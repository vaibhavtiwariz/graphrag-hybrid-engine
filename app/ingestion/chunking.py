from __future__ import annotations

import re


def split_text(text: str, size: int = 800, overlap: int = 120) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    current = ""
    step = max(size - overlap, 64)

    for paragraph in paragraphs:
        if len(paragraph) > size:
            if current:
                chunks.append(current)
                current = ""
            pieces = [paragraph[i : i + size] for i in range(0, len(paragraph), step)]
            chunks.extend(pieces[:-1])
            current = pieces[-1]
        elif not current:
            current = paragraph
        elif len(current) + len(paragraph) + 2 <= size:
            current = f"{current}\n\n{paragraph}"
        else:
            chunks.append(current)
            current = paragraph

    if current:
        chunks.append(current)

    kept = [c.strip() for c in chunks if len(c.strip()) >= 60]
    if not kept:
        return [text[:size]] if text else []
    return _apply_overlap(kept, overlap)


def _apply_overlap(chunks: list[str], overlap: int) -> list[str]:
    """True sliding window: each chunk after the first starts with the tail of
    the previous chunk, breaking at a paragraph boundary inside the overlap
    region when one exists."""
    if overlap <= 0 or len(chunks) < 2:
        return chunks
    out = [chunks[0]]
    for nxt in chunks[1:]:
        tail = out[-1][-overlap:]
        cut = tail.rfind("\n\n")
        carry = tail[cut + 2 :] if cut != -1 else tail
        # Prepend the carry verbatim (no strip): stripping would eat a leading
        # space and break the exact overlap continuity the window promises.
        out.append(f"{carry}\n\n{nxt}" if carry else nxt)
    return out
