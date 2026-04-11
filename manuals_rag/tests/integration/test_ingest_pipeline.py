from tests.helpers import fixture_pdf_path


def test_fixture_pdf_exists():
    fixture = fixture_pdf_path()
    assert fixture.exists()
