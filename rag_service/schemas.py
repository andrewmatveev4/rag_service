from pydantic import BaseModel


class Document(BaseModel):
    source: str
    text: str
    raw_chars: int

    @property
    def chars(self) -> int:
        return len(self.text)


class Chunk(BaseModel):
    document_id: str
    source_name: str
    chunk_id: str
    chunk_index: int
    text: str

    @property
    def words(self) -> int:
        return len(self.text.split())
