"""Read-only preflight: python -m cloudops_hub.answer_readiness."""

import argparse
import json

from azure.identity import DefaultAzureCredential
from azure.search.documents import SearchClient

from cloudops_hub.answer_scope import routing_registry
from cloudops_hub.config import get_settings
from cloudops_hub.documented_answers import collect_answers, source_links
from cloudops_hub.retrieval import (
    ENRICHED_FIELDS,
    SEARCH_FIELDS,
    documentation_passage,
    search_scope,
)


def check_readiness(search, settings, *, assistant_id: str = "landing-zone") -> dict:
    if assistant_id not in {"landing-zone", "platform-admin"}:
        raise ValueError("Answer routing preflight supports documentation assistants only")
    scope = search_scope(
        assistant_id, knowledge_graph_enabled=settings.knowledge_graph_enabled,
        confluence_enabled=settings.confluence_retrieval_enabled,
    )
    records = routing_registry(search, scope)
    expected = {
        row["answer_id"] for row in records if row.get("answer_support") != "generic"
    }
    if not expected:
        raise RuntimeError("Answer routing registry has no complete resource-specific answers")
    documents = {
        (row.get("source_id"), row.get("source_path")): row for row in records
    }
    reconstructed = set()
    for hit in documents.values():
        if not all(hit.get(key) for key in ("source_id", "source_path", "doc_id", "version")):
            raise RuntimeError("Answer routing registry has incomplete source provenance")
        candidates = collect_answers(
            search, [{**hit, "answer_id": ""}], settings, scope,
            SEARCH_FIELDS + ENRICHED_FIELDS, contract_fields=True,
        )
        for candidate in candidates:
            source = candidate["source"]
            identifier = source.get("answer_id")
            if identifier not in expected:
                continue
            if (
                identifier in reconstructed
                or not isinstance(source.get("answer_section_body"), str)
                or source.get("answer_presentation_version") != 1
                or source.get("answer_disposition") != "direct"
            ):
                raise RuntimeError("Answer routing reconstruction is incomplete or ambiguous")
            _, passage = documentation_passage({
                **source,
                "content": source_links(candidate["answer"], source.get("source_url", "")),
            }, 1, assistant_id)
            if len(passage) > settings.retrieval_context_chars:
                raise RuntimeError("Complete answer and source metadata exceed the evidence budget")
            reconstructed.add(identifier)
    if reconstructed != expected:
        raise RuntimeError("Not every approved answer route reconstructs a complete answer")
    return {"ready": True, "approved_answers": len(expected), "documents": len(documents)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--assistant", choices=["landing-zone", "platform-admin"], default="landing-zone",
    )
    args = parser.parse_args()
    settings = get_settings().for_assistant(args.assistant)
    if not settings.azure_search_endpoint or not settings.azure_search_index:
        raise RuntimeError("Azure Search must be configured for answer routing preflight")
    with DefaultAzureCredential() as credential, SearchClient(
        settings.azure_search_endpoint, settings.azure_search_index, credential=credential,
    ) as search:
        print(json.dumps(check_readiness(search, settings, assistant_id=args.assistant)))


if __name__ == "__main__":
    main()
