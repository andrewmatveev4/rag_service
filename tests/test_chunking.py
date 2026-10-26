from rag_service.chunking import chunk_document, chunk_documents
from rag_service.schemas import Document


def make_doc(source: str, n_words: int) -> Document:
    text = " ".join(f"w{i}" for i in range(n_words))
    return Document(source=source, text=text, raw_chars=len(text))


def test_chunk_metadata():
    chunks = chunk_document(make_doc("02_discounts.txt", 91), 50, 10)
    assert [c.chunk_id for c in chunks] == ["02_discounts:0", "02_discounts:1", "02_discounts:2"]
    assert [c.chunk_index for c in chunks] == [0, 1, 2]
    assert {c.document_id for c in chunks} == {"02_discounts"}
    assert {c.source_name for c in chunks} == {"02_discounts.txt"}


def test_chunk_ids_are_stable_between_runs():
    doc = make_doc("a.txt", 120)
    assert [c.chunk_id for c in chunk_document(doc)] == [c.chunk_id for c in chunk_document(doc)]


def test_chunk_ids_unique_across_documents():
    chunks = chunk_documents([make_doc("a.txt", 120), make_doc("b.txt", 120)])
    ids = [c.chunk_id for c in chunks]
    assert len(chunks) == 6
    assert len(ids) == len(set(ids))
