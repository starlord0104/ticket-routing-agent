from src.preprocess import clean_text, map_label


def test_clean_text_removes_html_urls_email_ticket_ids():
    text = '<b>Hello</b> INC0012345 https://example.com test@example.com!'
    out = clean_text(text)

    assert '<b>' not in out
    assert 'inc0012345' not in out
    assert 'example.com' not in out
    assert 'test@example.com' not in out
    assert out == 'hello'


def test_map_label_unknown_returns_none():
    assert map_label('not-a-real-label') is None


def test_map_label_access_merge():
    assert map_label('Administrative rights') == 'Access Management'
    assert map_label('access') == 'Access Management'