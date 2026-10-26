import pandas as pd

from nowcast import releases
from nowcast.releases import build_releases, html_to_text

CIK = 21344
RECENT = {
    "accessionNumber": ["0000021344-24-000010", "0000021344-24-000011", "0000021344-24-000012"],
    "filingDate": ["2024-02-13", "2024-03-01", "2024-04-30"],
    "acceptanceDateTime": ["2024-02-13T06:55:10.000Z", "2024-03-01T16:00:00.000Z", "2024-04-30T21:40:00.000Z"],
    "form": ["8-K", "10-K", "8-K"],
    "items": ["2.02,9.01", "", "2.02,7.01,9.01"],
}
OLD = {
    "accessionNumber": ["0000021344-15-000001", "0000021344-14-000001"],
    "filingDate": ["2015-07-22", "2014-07-22"],
    "acceptanceDateTime": ["2015-07-22T11:00:00.000Z", "2014-07-22T11:00:00.000Z"],
    "form": ["8-K", "8-K"],
    "items": ["2.02", "2.02"],
}
INDEX = """<table summary="Document Format Files">
<tr><th>Seq</th><th>Description</th><th>Document</th><th>Type</th><th>Size</th></tr>
<tr><td>1</td><td>8-K</td><td><a href="/ix?doc=/Archives/edgar/data/21344/x/ko-8k.htm">ko-8k.htm</a></td><td>8-K</td><td>50</td></tr>
<tr><td>2</td><td>PRESS RELEASE</td><td><a href="/Archives/edgar/data/21344/x/ex991.htm">ex991.htm</a></td><td>EX-99.1</td><td>90</td></tr>
</table>"""
EXHIBIT = """<html><head><style>p{}</style></head><body>
<p>Coca-Cola Reports First Quarter Results</p>
<p>The company   expects full-year
organic revenue growth of 8% to 9%.</p><script>x()</script></body></html>"""


class FakeClient:
    def __init__(self):
        self.urls = []

    def get_json(self, url, cache_key=None, refresh=False):
        self.urls.append(url)
        if url.endswith(f"CIK{CIK:010d}.json"):
            return {"filings": {"recent": RECENT, "files": [
                {"name": "CIK0000021344-submissions-001.json", "filingTo": "2015-12-31"},
                {"name": "CIK0000021344-submissions-002.json", "filingTo": "2010-12-31"},
            ]}}
        if url.endswith("submissions-001.json"):
            return OLD
        raise AssertionError(f"page outside the window was requested: {url}")

    def get_text(self, url, cache_key=None, refresh=False):
        self.urls.append(url)
        return INDEX if url.endswith("-index.htm") else EXHIBIT


def test_build_releases(tmp_path, monkeypatch):
    monkeypatch.setattr(releases, "TEXT_DIR", tmp_path)
    universe = pd.DataFrame({"ticker": ["KO"], "cik": [CIK]})
    client = FakeClient()
    df, missing = build_releases(client, universe)

    assert missing == []
    assert list(df["accession"]) == ["0000021344-15-000001", "0000021344-24-000010", "0000021344-24-000012"]
    assert df["available_at"].tolist() == [pd.Timestamp("2015-07-23"), pd.Timestamp("2024-02-14"),
                                           pd.Timestamp("2024-05-01")]  # day after acceptance
    assert "https://www.sec.gov/Archives/edgar/data/21344/x/ex991.htm" in client.urls  # exhibit, not the 8-K
    text = open(df["path"].iloc[0]).read()
    assert "expects full-year organic revenue growth of 8% to 9%." in text.replace("\n", " ")
    assert "x()" not in text


def test_html_to_text_passes_plain_text_through():
    assert html_to_text("plain release text") == "plain release text"


def test_html_to_text_handles_xml_declaration():
    raw = '<?xml version="1.0" encoding="ISO-8859-1"?>\n<html><body><p>Revenue rose 5%</p></body></html>'
    assert html_to_text(raw) == "Revenue rose 5%"


def test_bad_exhibit_is_logged_not_fatal(tmp_path, monkeypatch):
    monkeypatch.setattr(releases, "TEXT_DIR", tmp_path)

    class EmptyExhibit(FakeClient):
        def get_text(self, url, cache_key=None, refresh=False):
            return INDEX if url.endswith("-index.htm") else "<html></html>"

    df, missing = build_releases(EmptyExhibit(), pd.DataFrame({"ticker": ["KO"], "cik": [CIK]}))
    assert df.empty and len(missing) == 3
