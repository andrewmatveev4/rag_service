import pytest

from rag_service.loader import load_documents


def test_reads_only_txt_sorted(tmp_path):
    (tmp_path / "b.txt").write_text("второй", encoding="utf-8")
    (tmp_path / "a.txt").write_text("первый", encoding="utf-8")
    (tmp_path / "notes.md").write_text("не документ", encoding="utf-8")
    docs = load_documents(str(tmp_path))
    assert [d.source for d in docs] == ["a.txt", "b.txt"]


def test_text_is_cleaned_and_raw_length_kept(tmp_path):
    (tmp_path / "a.txt").write_text("  скидка   15%  \n\n\n", encoding="utf-8")
    doc = load_documents(str(tmp_path))[0]
    assert doc.text == "скидка 15%"
    assert doc.raw_chars == 19
    assert doc.chars == 10


def test_bom_removed(tmp_path):
    (tmp_path / "a.txt").write_bytes("\ufeffВозврат".encode("utf-8"))
    assert load_documents(str(tmp_path))[0].text == "Возврат"


def test_empty_document_skipped(tmp_path):
    (tmp_path / "a.txt").write_text(" \n\t\n", encoding="utf-8")
    (tmp_path / "b.txt").write_text("текст", encoding="utf-8")
    assert [d.source for d in load_documents(str(tmp_path))] == ["b.txt"]


def test_missing_folder_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_documents(str(tmp_path / "nope"))
