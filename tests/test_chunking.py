from __future__ import annotations

from app.ingestion.chunking import split_text


def test_paragraphs_merged_within_size_limit():
    text = "\n\n".join(f"para {i} " + "word " * 20 for i in range(6))
    chunks = split_text(text, size=400, overlap=80)
    assert len(chunks) >= 1
    # chunks after the first carry up to `overlap` chars of sliding-window context
    assert all(len(c) <= 400 + 80 for c in chunks)


def test_long_paragraph_is_hard_split_with_overlap():
    paragraph = "sentence. " * 300
    chunks = split_text(paragraph, size=500, overlap=100)
    assert len(chunks) >= 3
    assert chunks[0][-30:] in chunks[0] + chunks[1]


def test_short_noise_filtered_out():
    text = "tiny\n\nok this one is long enough to survive filtering because it has many words here"
    chunks = split_text(text, size=800, overlap=120)
    assert all(len(c) >= 60 for c in chunks)


def test_no_text_returns_empty():
    assert split_text("", size=800, overlap=120) == []


def test_single_small_document_round_trips():
    body = "A" * 300
    chunks = split_text(body, size=800, overlap=120)
    assert chunks == [body]


def test_sliding_window_overlap_carries_tail():
    text = "word " * 400  # one long paragraph, no blank lines inside
    chunks = split_text(text, size=200, overlap=50)
    assert len(chunks) >= 2
    assert chunks[1].startswith(chunks[0][-50:])


def test_overlap_prefers_paragraph_boundary():
    text = "\n\n".join(f"para {i} " + "x" * 150 for i in range(4))
    chunks = split_text(text, size=250, overlap=100)
    assert len(chunks) >= 2
    # the carried prefix must be a suffix fragment of the previous chunk
    assert chunks[1][:100] in chunks[0] or chunks[0][-100:] in chunks[1]
