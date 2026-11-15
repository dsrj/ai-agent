"""Verify the pinned source ontology and validate its graph before evidence integration."""

import hashlib
import json
from pathlib import Path
from typing import Any

from pyshacl import validate
from rdflib import OWL, RDF, Graph, Namespace

VOCABULARY = Namespace(
    "https://github.com/almbrand/cloudops-openai-hub-knowledge-source/knowledge-graph/vocab/"
)


def _contains_external_context(value: Any) -> bool:
    if isinstance(value, dict):
        return any(
            key in {"@context", "@import"} or _contains_external_context(child)
            for key, child in value.items()
        )
    if isinstance(value, list):
        return any(_contains_external_context(child) for child in value)
    return False


def load_validated_schema(
    root: Path, snapshot: Any, *, validate_graph: bool = True,
) -> dict[str, Any]:
    model = root / "model"
    manifest = json.loads((model / "manifest.json").read_bytes())
    names = {"ontology.ttl", "shapes.ttl", "context.json"}
    if (
        manifest.get("source_commit") != snapshot.commit
        or manifest.get("snapshot_sha256") != snapshot.digest
        or set(manifest.get("files", {})) != names
    ):
        raise ValueError("Graph model does not match the approved snapshot")
    content = {}
    for name in names:
        content[name] = (model / name).read_bytes()
        if hashlib.sha256(content[name]).hexdigest() != manifest["files"][name]:
            raise ValueError("Graph model checksum mismatch")
    context = json.loads(content["context.json"])
    if (
        not isinstance(context, dict) or context.get("kg") != str(VOCABULARY)
        or _contains_external_context(context)
        or _contains_external_context(list(snapshot.records.values()))
    ):
        raise ValueError("Graph JSON-LD context is invalid")
    ontology = Graph().parse(data=content["ontology.ttl"], format="turtle")
    shapes = Graph().parse(data=content["shapes.ttl"], format="turtle")
    classes = sorted(
        str(value)[len(str(VOCABULARY)):]
        for value in ontology.subjects(RDF.type, OWL.Class)
        if str(value).startswith(str(VOCABULARY))
    )
    properties = sorted({
        str(value)[len(str(VOCABULARY)):]
        for property_type in (OWL.ObjectProperty, OWL.DatatypeProperty)
        for value in ontology.subjects(RDF.type, property_type)
        if str(value).startswith(str(VOCABULARY))
    })
    if not {"Entity", "Claim", "Relationship", "Evidence"}.issubset(classes):
        raise ValueError("Graph ontology is missing required classes")
    if not {"subject", "object", "source", "predicate"}.issubset(properties):
        raise ValueError("Graph ontology is missing required properties")
    inverses = {
        str(subject)[len(str(VOCABULARY)):]: str(target)[len(str(VOCABULARY)):]
        for subject, target in ontology.subject_objects(OWL.inverseOf)
        if str(subject).startswith(str(VOCABULARY))
        and str(target).startswith(str(VOCABULARY))
    }
    if validate_graph:
        data = Graph().parse(
            data=json.dumps({"@context": context, "@graph": list(snapshot.records.values())}),
            format="json-ld",
        )
        conforms, _, _ = validate(data, shacl_graph=shapes, ont_graph=ontology, inplace=True)
        if not conforms:
            raise ValueError("Approved graph fails SHACL validation")
    return {"classes": classes, "properties": properties, "inverses": inverses}
