import logging

from rag_service.chunking import chunk_documents
from rag_service.embedder import embed_texts
from rag_service.loader import load_documents
from rag_service.weaviate_store import (
    chunk_uuid,
    connect,
    count_objects,
    delete_stale,
    ensure_collection,
    index_chunks,
)

logger = logging.getLogger(__name__)
KNOWLEDGE_BASE_DIR = "knowledge_base"


def index_knowledge_base(folder: str = KNOWLEDGE_BASE_DIR) -> dict:
    docs = load_documents(folder)
    chunks = chunk_documents(docs)
    vectors = embed_texts([c.text for c in chunks])
    with connect() as client:
        collection = ensure_collection(client)
        loaded = index_chunks(collection, chunks, vectors)
        deleted = delete_stale(collection, {chunk_uuid(c.chunk_id) for c in chunks})
        total = count_objects(collection)
    return {"documents": len(docs), "chunks": len(chunks), "loaded": loaded,
            "deleted_stale": deleted, "total_in_db": total}


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("rag_service").setLevel(logging.INFO)
    print(index_knowledge_base())
