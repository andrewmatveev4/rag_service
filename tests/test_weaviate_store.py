from rag_service.schemas import Chunk
from rag_service.weaviate_store import chunk_properties, chunk_uuid


def make_chunk(chunk_id: str = "02_discounts:1") -> Chunk:
    document_id, index = chunk_id.split(":")
    return Chunk(document_id=document_id, source_name=f"{document_id}.txt",
                 chunk_id=chunk_id, chunk_index=int(index), text="Скидок нет.")


def test_uuid_is_stable():
    assert chunk_uuid("02_discounts:1") == chunk_uuid("02_discounts:1")


def test_uuid_differs_between_chunks():
    assert chunk_uuid("02_discounts:1") != chunk_uuid("02_discounts:2")


def test_properties_keep_all_metadata():
    props = chunk_properties(make_chunk())
    assert props == {
        "document_id": "02_discounts",
        "source_name": "02_discounts.txt",
        "chunk_id": "02_discounts:1",
        "chunk_index": 1,
        "text": "Скидок нет.",
    }
