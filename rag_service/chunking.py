from pathlib import Path

from rag_service.schemas import Chunk, Document
from rag_service.loader import load_documents

CHUNK_SIZE_WORDS = 50
CHUNK_OVERLAP_WORDS = 10


def split_text(text: str, size: int, overlap: int) -> list[str]:
    words = text.split()
    step = size - overlap
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError(f"Invalid size/overlap: {size}/{overlap}")
    chunks = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start:start + size]))
        if start + size >= len(words):
            break
    return chunks


def chunk_document(
    doc: Document, size: int = CHUNK_SIZE_WORDS, overlap: int = CHUNK_OVERLAP_WORDS
) -> list[Chunk]:
    document_id = Path(doc.source).stem
    return [
        Chunk(
            document_id=document_id,
            source_name=doc.source,
            chunk_id=f"{document_id}:{index}",
            chunk_index=index,
            text=text,
        )
        for index, text in enumerate(split_text(doc.text, size, overlap))
    ]


def chunk_documents(
    docs: list[Document], size: int = CHUNK_SIZE_WORDS, overlap: int = CHUNK_OVERLAP_WORDS
) -> list[Chunk]:
    return [chunk for doc in docs for chunk in chunk_document(doc, size, overlap)]


if __name__ == "__main__":
    docs = load_documents("knowledge_base")
    chunks = chunk_documents(docs)
    print(f"size={CHUNK_SIZE_WORDS}, overlap={CHUNK_OVERLAP_WORDS}: "
          f"{len(docs)} документов, {len(chunks)} чанков")
    for doc in docs:
        sizes = [c.words for c in chunks if c.source_name == doc.source]
        print(f"{doc.source:<36}{len(doc.text.split()):>4} слов -> {sizes}")
    for chunk in chunks[3:6]:
        print()
        print(chunk.model_dump(exclude={"text"}), chunk.words, "слов")
        print(chunk.text)
