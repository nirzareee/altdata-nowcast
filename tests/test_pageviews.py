import pandas as pd

from nowcast.pageviews import build_pageviews


class FakeClient:
    def __init__(self):
        self.urls = []

    def get_json(self, url, cache_key=None, refresh=False):
        self.urls.append(url)
        if "Missing_Page" in url:
            return None
        return {"items": [
            {"article": "x", "timestamp": "2024010100", "views": 10},
            {"article": "x", "timestamp": "2024010200", "views": 12},
        ]}


def test_build_pageviews():
    universe = pd.DataFrame({"ticker": ["MCD", "ZZZ"], "wiki_articles": ["McDonald's", "Missing_Page"]})
    client = FakeClient()
    df, missing = build_pageviews(client, universe, "2010-01-01", "2024-01-02")

    assert missing == [("ZZZ", "Missing_Page")]
    assert list(df["views"]) == [10, 12]
    assert (df["available_at"] - df["date"]).eq(pd.Timedelta(days=1)).all()
    assert "McDonald%27s" in client.urls[0]
    assert "/20150701" in client.urls[0]  # clamped to first available day
