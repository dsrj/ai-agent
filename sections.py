"""Section-aware Markdown/text chunks with repeated heading and table context."""

import re


def _blocks(lines: list[str]) -> list[str]:
    blocks = []
    current = []
    fence = None
    for line in lines:
        marker = re.match(r"^\s*(`{3,}|~{3,})", line)
        if marker:
            if fence is None:
                if current:
                    blocks.append("".join(current))
                    current = []
                fence = marker[1][0]
            elif marker[1][0] == fence:
                current.append(line)
                blocks.append("".join(current))
                current = []
                fence = None
                continue
        if not line.strip() and fence is None:
            if current:
                blocks.append("".join(current))
                current = []
        else:
            current.append(line)
    if current:
        blocks.append("".join(current))
    return blocks


def section_chunks(text: str, max_chars: int = 6000) -> list[tuple[str, str]]:
    return [
        (heading, content)
        for heading, content, _ in section_chunks_with_bodies(text, max_chars)
    ]


def section_chunks_with_bodies(
    text: str, max_chars: int = 6000,
) -> list[tuple[str, str, str]]:
    lines = text.splitlines(keepends=True)
    sections: list[tuple[str, list[str]]] = []
    headings: list[str] = []
    body: list[str] = []
    fence = None
    for line in lines:
        marker = re.match(r"^\s*(`{3,}|~{3,})", line)
        heading = re.match(r"^(#{1,6})\s+(.+?)\s*#*\s*$", line) if fence is None else None
        if heading:
            if body:
                sections.append((" > ".join(headings), body))
            level = len(heading[1])
            headings = headings[: level - 1] + [heading[2]]
            body = []
        else:
            body.append(line)
        if marker:
            if fence is None:
                fence = marker[1][0]
            elif marker[1][0] == fence:
                fence = None
    if body or headings:
        sections.append((" > ".join(headings), body))

    result = []
    for heading, content in sections:
        prefix = f"## {heading}\n\n" if heading else ""
        original_body = "".join(content).strip()

        def emit(value, section=heading):
            nonlocal original_body
            result.append((section, value, original_body))
            original_body = ""

        budget = max_chars - len(prefix)
        if budget < 100:
            raise ValueError("Section heading exceeds chunk budget")
        blocks = _blocks(content)
        current = ""
        for block in blocks:
            if not block:
                continue
            if current and len(current) + len(block) + 2 > budget:
                emit(prefix + current.strip())
                current = ""
            # Long blocks split at line boundaries. Repeat code fences/table headers.
            if len(block) > budget:
                rows = block.splitlines(keepends=True)
                code = (
                    rows[0].rstrip() if rows and rows[0].lstrip().startswith(("```", "~~~")) else ""
                )
                table = (
                    "".join(rows[:2])
                    if len(rows) > 1 and "|" in rows[0] and re.search(r"\|?\s*:?-{3,}", rows[1])
                    else ""
                )
                repeat = f"{code}\n" if code else table
                closing = "\n" + code[:3] if code else ""
                if code:
                    rows = rows[1:-1] if rows[-1].strip().startswith(code[:3]) else rows[1:]
                elif table:
                    rows = rows[2:]
                piece = repeat
                for row in rows:
                    room = budget - len(repeat) - len(closing) - 1
                    if len(row) > room:
                        if room < 100:
                            raise ValueError("A code/table header exceeds the chunk budget")
                        # Oversized rows are hard-split; each part keeps the fence/table header.
                        if piece.strip() and piece != repeat:
                            emit(prefix + piece.strip() + closing)
                        for start in range(0, len(row), room):
                            part = row[start : start + room].strip()
                            emit(prefix + (repeat + part).strip() + closing)
                        piece = repeat
                        continue
                    if len(piece) + len(row) + len(closing) > budget:
                        emit(prefix + piece.strip() + closing)
                        piece = repeat
                    piece += row
                if piece.strip():
                    emit(prefix + piece.strip() + closing)
            else:
                current = f"{current}\n\n{block}".strip()
        if current.strip():
            emit(prefix + current.strip())
    return result
