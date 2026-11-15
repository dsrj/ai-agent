"""Package the reviewed CloudOps Azure Firewall standards as a Firewall knowledge source."""

import hashlib
from pathlib import Path

from cloudops_hub.ingest import ingest

GUIDE_DIR = Path(__file__).resolve().parent.parent / "data" / "firewall-guide"
SOURCE_ID = "guide:firewall-standards"
GUIDE_FILES = ("internal-destinations.md",)
SOURCE_BASE = (
    "https://github.com/almbrand/cloudops-openai-hub/blob/main/backend/data/firewall-guide"
)


def ingest_firewall_guide(*, before_write=None) -> None:
    metadata = {}
    digest = hashlib.sha256()
    for filename in GUIDE_FILES:
        text = (GUIDE_DIR / filename).read_text(encoding="utf-8").replace("\r\n", "\n")
        if not text.strip():
            raise ValueError(f"Firewall guide document is empty: {filename}")
        digest.update(filename.encode() + b"\0" + text.encode("utf-8") + b"\0")
        metadata[filename] = {
            "title": text.splitlines()[0].lstrip("# "),
            "url": f"{SOURCE_BASE}/{filename}",
            "assistant": "firewall",
        }
    actual = {path.name for path in GUIDE_DIR.iterdir()}
    if actual != set(GUIDE_FILES):
        raise ValueError("Firewall guide contains unreviewed or missing documents")
    ingest(
        GUIDE_DIR,
        source_id=SOURCE_ID,
        version=digest.hexdigest(),
        source_kind="operations",
        assistant="firewall",
        metadata=metadata,
        before_write=before_write,
    )
