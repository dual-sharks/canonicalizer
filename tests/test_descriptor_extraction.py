import pytest

spacy = pytest.importorskip("spacy")
from services.canonicalizer import Canonicalizer


def test_extract_descriptors_no_conjunction():
    nlp = spacy.load("en_core_web_sm")
    canon = Canonicalizer.__new__(Canonicalizer)
    text = "This is a test. The graphics are absolutely amazing."
    doc = nlp(text)
    match_token = next(t for t in doc if t.text.lower() == "graphics")
    match_span = doc[match_token.i : match_token.i + 1]
    sent = match_span.sent
    result = canon._extract_descriptors(sent, match_span)
    assert result == "absolutely amazing"
