from pydantic import BaseModel


class Document(BaseModel):
    source: str
    text: str
    raw_chars: int

    @property
    def chars(self) -> int:
        return len(self.text)
