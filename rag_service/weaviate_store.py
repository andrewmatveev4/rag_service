import logging

import weaviate
from weaviate import WeaviateClient
from weaviate.classes.config import Configure, DataType, Property, Tokenization
from weaviate.classes.query import Filter
from weaviate.collections import Collection
from weaviate.util import generate_uuid5

from rag_service.schemas import Chunk

logger = logging.getLogger(__name__)
COLLECTION_NAME = "KnowledgeChunk"


class StoreError(Exception):
    pass


def connect() -> WeaviateClient:
    return weaviate.connect_to_local()


def ensure_collection(client: WeaviateClient) -> Collection:
    if not client.collections.exists(COLLECTION_NAME):
        client.collections.create(
            name=COLLECTION_NAME,
            vector_config=Configure.Vectors.self_provided(),
            properties=[
                Property(name="document_id", data_type=DataType.TEXT,
                         tokenization=Tokenization.FIELD),
                Property(name="source_name", data_type=DataType.TEXT,
                         tokenization=Tokenization.FIELD),
                Property(name="chunk_id", data_type=DataType.TEXT,
                         tokenization=Tokenization.FIELD),
                Property(name="chunk_index", data_type=DataType.INT),
                Property(name="text", data_type=DataType.TEXT),
            ],
        )
        logger.info("Создана collection %s", COLLECTION_NAME)
    return client.collections.use(COLLECTION_NAME)


def chunk_uuid(chunk_id: str) -> str:
    return generate_uuid5(chunk_id)


def chunk_properties(chunk: Chunk) -> dict:
    return chunk.model_dump()


def index_chunks(collection: Collection, chunks: list[Chunk], vectors: list[list[float]]) -> int:
    if len(chunks) != len(vectors):
        raise ValueError(f"Чанков {len(chunks)}, а векторов {len(vectors)}")
    with collection.batch.fixed_size(batch_size=100) as batch:
        for chunk, vector in zip(chunks, vectors):
            batch.add_object(
                properties=chunk_properties(chunk),
                uuid=chunk_uuid(chunk.chunk_id),
                vector=vector,
            )
    failed = collection.batch.failed_objects
    if failed:
        raise StoreError(f"Не загружено {len(failed)} объектов: {failed[0].message}")
    return len(chunks)


def delete_stale(collection: Collection, keep_uuids: set[str]) -> int:
    existing = {str(obj.uuid) for obj in collection.iterator()}
    stale = existing - keep_uuids
    if stale:
        collection.data.delete_many(where=Filter.by_id().contains_any(list(stale)))
    return len(stale)


def count_objects(collection: Collection) -> int:
    return collection.aggregate.over_all(total_count=True).total_count
