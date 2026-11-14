FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY cloudops_hub ./cloudops_hub
COPY data ./data
COPY .graph-build ./graph
ENV PYTHONPATH=/app/graph GRAPH_SNAPSHOT_DIRECTORY=/app/graph
RUN python -c 'import json; from pathlib import Path; from cloudops_knowledge_mcp.snapshot import parse_snapshot; from cloudops_hub.graph_schema import load_validated_schema; p = Path("/app/graph"); pin = json.loads((p / "snapshot-pin.json").read_bytes()); s = parse_snapshot((p / "snapshot.json").read_bytes(), pin["snapshot_sha256"]); assert (s.commit, s.manifest_sha256) == (pin["commit"], pin["manifest_sha256"]); load_validated_schema(p, s)'
RUN useradd --create-home --uid 10001 appuser
USER 10001
EXPOSE 8000
CMD ["uvicorn", "cloudops_hub.main:app", "--host", "0.0.0.0", "--port", "8000"]
