import logging
import re
from pathlib import Path

from rag_service.schemas import Document

logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    lines = [" ".join(line.split()) for line in text.splitlines()]
    joined = "\n".join(lines)
    return re.sub(r"\n{2,}", "\n\n", joined).strip()


def load_documents(folder: str) -> list[Document]:
    root = Path(folder)
    if not root.is_dir():
        raise FileNotFoundError(f"Папка с базой знаний не найдена: {root}")
    docs = []
    for path in sorted(root.glob("*.txt")):
        raw = path.read_text(encoding="utf-8-sig")
        text = clean_text(raw)
        if not text:
            logger.warning("Пустой документ пропущен: %s", path.name)
            continue
        docs.append(Document(source=path.name, text=text, raw_chars=len(raw)))
    return docs


if __name__ == "__main__":
    docs = load_documents("knowledge_base")
    print(f"{'документ':<36}{'было':>7}{'стало':>7}{'убрано':>8}")
    for doc in docs:
        print(f"{doc.source:<36}{doc.raw_chars:>7}{doc.chars:>7}{doc.raw_chars - doc.chars:>8}")
    raw_total = sum(d.raw_chars for d in docs)
    total = sum(d.chars for d in docs)
    print(f"{'итого: ' + str(len(docs)):<36}{raw_total:>7}{total:>7}{raw_total - total:>8}")
    print(repr(docs[1].text[:150]))
