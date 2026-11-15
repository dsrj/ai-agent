"""Repeatable local retrieval proxy and opt-in Azure answer capture for human review."""

import argparse
import json
import math
import re
from pathlib import Path

from cloudops_hub.config import Settings, get_settings
from cloudops_hub.knowledge import expand_query, identifiers, load_synonyms

DEFAULT_CASES = Path(__file__).resolve().parents[1] / "evaluation" / "retrieval-cases.json"
FIREWALL_CASES = Path(__file__).resolve().parents[1] / "evaluation" / "firewall-intent-cases.json"


def tokens(text):
    return re.findall(r"[\w]+", text.casefold())


def rank_local(documents, question, query, assistant_id, improved):
    candidates = [doc for doc in documents if doc["assistant"] in {assistant_id, "shared"}]
    terms = set(tokens(query))
    exact = identifiers(question)
    scored = []
    for doc in candidates:
        body = tokens(doc["title"] + " " + doc["content"])
        score = 0.0
        for term in terms:
            frequency = body.count(term)
            count = sum(
                term in tokens(item["title"] + " " + item["content"]) for item in candidates
            )
            idf = math.log(1 + (len(candidates) - count + 0.5) / (count + 0.5))
            score += idf * frequency * 2.2 / (frequency + 1.2 * (0.25 + 0.75 * len(body) / 30))
        if improved and any(term in doc["content"] for term in exact):
            score += 100
        if score:
            scored.append((score, doc["id"]))
    return [key for _, key in sorted(scored, reverse=True)[:3]]


def retrieval_metrics(rows):
    answerable = [row for row in rows if row["relevant"]]
    if not answerable:
        raise ValueError("Evaluation needs at least one labelled answerable question")
    return {
        "answerable_cases": len(answerable),
        "recall_at_3": sum(
            len(set(row["retrieved"]) & set(row["relevant"])) / len(row["relevant"])
            for row in answerable
        )
        / len(answerable),
        "mrr_at_3": sum(
            next(
                (1 / (i + 1) for i, key in enumerate(row["retrieved"]) if key in row["relevant"]), 0
            )
            for row in answerable
        )
        / len(answerable),
        "source_precision_at_3": sum(
            len(set(row["retrieved"]) & set(row["relevant"])) / max(1, len(row["retrieved"]))
            for row in answerable
        )
        / len(answerable),
    }


def evaluate_local(dataset, settings):
    synonyms = load_synonyms(settings)
    report = {
        "kind": "synthetic local BM25-style proxy, not Azure hybrid or model answer accuracy",
        "answer_accuracy": None,
        "source_correctness": "relevance labels only; real citations require administrator review",
    }
    for mode in ("before", "after"):
        rows = []
        for case in dataset["questions"]:
            question = case["question"]
            query = (
                question
                if mode == "before"
                else expand_query(question, case.get("interpretation", question), synonyms)
            )
            rows.append(
                {
                    "id": case["id"],
                    "relevant": case["relevant"],
                    "retrieved": rank_local(
                        dataset["documents"], question, query, case["assistant"], mode == "after"
                    ),
                }
            )
        report[mode] = {**retrieval_metrics(rows), "cases": rows}
    return report


def review_metrics(rows):
    reviewed = [row for row in rows if isinstance(row.get("accurate"), bool)]
    return {
        "reviewed": len(reviewed),
        "answer_accuracy": sum(row["accurate"] for row in reviewed) / len(reviewed)
        if reviewed
        else None,
        "source_correctness": (
            sum(row.get("sources_correct") is True for row in reviewed) / len(reviewed)
            if reviewed
            else None
        ),
    }


def capture_firewall_intents(dataset, settings, understand=None):
    """Measure actual route/field predictions, without executing any live Firewall tools."""
    from cloudops_hub.firewall_json import _firewall_json_chat
    from cloudops_hub.firewall_model_data import firewall_boundary

    understand = understand or _firewall_json_chat
    rows = []
    for case in dataset["questions"]:
        with firewall_boundary():
            result = (
                understand(
                    settings,
                    case["question"],
                    case.get("history", []),
                    analysis_state=case.get("analysis_state"),
                )
                or {}
            )
        actual = {
            "intent": result.get("intent"),
            "task": result.get("task"),
            **(result.get("understanding") or {}),
            "clarification": result.get("status") == "clarification",
        }
        matches = {key: actual.get(key) == value for key, value in case["expected"].items()}
        rows.append(
            {
                "id": case["id"],
                "expected": case["expected"],
                "actual": actual,
                "field_matches": matches,
                "passed": all(matches.values()),
            }
        )
    if not rows:
        raise ValueError("Firewall intent evaluation requires labelled questions")
    matched = sum(sum(row["field_matches"].values()) for row in rows)
    total = sum(len(row["field_matches"]) for row in rows)
    return {
        "kind": "model intent/field evaluation only; not live query or answer accuracy",
        "case_accuracy": sum(row["passed"] for row in rows) / len(rows),
        "field_accuracy": matched / total if total else None,
        "cases": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path)
    parser.add_argument(
        "--capture-live",
        action="store_true",
        help="Send curated questions to configured Azure services; no index writes",
    )
    parser.add_argument(
        "--review", type=Path, help="Summarise administrator-labelled answer capture"
    )
    parser.add_argument(
        "--capture-firewall-intents",
        action="store_true",
        help="Evaluate Firewall routes and extracted fields on the configured model; "
        "no live Azure Firewall reads or writes",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.capture_firewall_intents and (args.capture_live or args.review):
        parser.error("Choose only one capture/review mode")
    path = args.cases or (FIREWALL_CASES if args.capture_firewall_intents else DEFAULT_CASES)
    dataset = json.loads(path.read_text(encoding="utf-8"))
    if args.review:
        report = review_metrics(json.loads(args.review.read_text(encoding="utf-8"))["cases"])
    elif args.capture_firewall_intents:
        report = capture_firewall_intents(dataset, get_settings())
    elif args.capture_live:
        from cloudops_hub.firewall_model_data import firewall_boundary, internal_boundary
        from cloudops_hub.retrieval import answer_question

        settings = get_settings()
        rows = []
        for case in dataset["questions"]:
            boundary = (
                firewall_boundary()
                if case["assistant"] == "firewall"
                else internal_boundary(settings)
            )
            with boundary as references:
                result = answer_question(
                    settings, case["assistant"], case["question"], case.get("history", [])
                )
                if references is not None and not references.restore_outputs:
                    result["answer"] = references.render(result["answer"])
            rows.append(
                {
                    **case,
                    "result": result,
                    "accurate": None,
                    "sources_correct": None,
                    "reviewer": "",
                    "review_notes": "",
                }
            )
        report = {"kind": "live capture; pending administrator review", "cases": rows}
    else:
        report = evaluate_local(dataset, Settings())
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    summary = {
        key: {field: metric for field, metric in value.items() if field != "cases"}
        if isinstance(value, dict)
        else value
        for key, value in report.items()
        if key != "cases"
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
