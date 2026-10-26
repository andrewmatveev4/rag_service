import pytest

from rag_service.chunking import split_text


def words(n: int) -> str:
    return " ".join(f"w{i}" for i in range(n))


def test_empty_text_gives_no_chunks():
    assert split_text("", 5, 1) == []
    assert split_text("  \n\n ", 5, 1) == []


def test_short_text_is_one_chunk():
    assert split_text("а б в", 5, 1) == ["а б в"]


def test_newlines_are_not_words():
    assert split_text("а\n\nб  в", 2, 0) == ["а б", "в"]


def test_windows_without_overlap():
    assert split_text(words(6), 3, 0) == ["w0 w1 w2", "w3 w4 w5"]


def test_windows_with_overlap():
    assert split_text(words(7), 4, 2) == ["w0 w1 w2 w3", "w2 w3 w4 w5", "w4 w5 w6"]


def test_exact_size_is_one_chunk():
    assert split_text(words(5), 5, 2) == [words(5)]


def test_no_tail_duplicate():
    assert split_text(words(6), 4, 2) == ["w0 w1 w2 w3", "w2 w3 w4 w5"]


def test_long_text_is_fully_covered():
    chunks = split_text(words(500), 50, 10)
    assert len(chunks) == 13
    assert all(len(c.split()) <= 50 for c in chunks)
    assert chunks[0].split()[0] == "w0"
    assert chunks[-1].split()[-1] == "w499"


def test_neighbours_share_overlap():
    chunks = split_text(words(500), 50, 10)
    for left, right in zip(chunks, chunks[1:]):
        assert left.split()[-10:] == right.split()[:10]


@pytest.mark.parametrize("size, overlap", [(0, 0), (5, 5), (5, 7), (5, -1)])
def test_invalid_params_raise(size, overlap):
    with pytest.raises(ValueError):
        split_text(words(10), size, overlap)
