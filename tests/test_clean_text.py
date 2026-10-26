from rag_service.loader import clean_text


def test_collapses_spaces_inside_line():
    assert clean_text("скидка 15%   на тариф") == "скидка 15% на тариф"


def test_tabs_and_nbsp_become_one_space():
    assert clean_text("490\xa0₽\tза\t\tучастника") == "490 ₽ за участника"


def test_strips_line_edges():
    assert clean_text("   первая  \n\tвторая\t") == "первая\nвторая"


def test_whitespace_only_line_becomes_empty():
    assert clean_text("а\n \t \nб") == "а\n\nб"


def test_single_newline_kept():
    assert clean_text("а\nб") == "а\nб"


def test_many_empty_lines_become_one_paragraph_break():
    assert clean_text("а\n\n\n\nб") == "а\n\nб"


def test_windows_newlines():
    assert clean_text("а\r\nб\r\n\r\n\r\nв") == "а\nб\n\nв"


def test_strips_text_edges():
    assert clean_text("\n\n  а  \n\n") == "а"


def test_empty_and_whitespace_only_input():
    assert clean_text("") == ""
    assert clean_text(" \n\t\xa0\n") == ""
