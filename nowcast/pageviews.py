"""Daily Wikipedia pageviews (human traffic only). Data starts 2015-07-01."""
from urllib.parse import quote

import pandas as pd

URL = (
    "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
    "en.wikipedia/all-access/user/{article}/daily/{start}/{end}"
)
FIRST_DAY = pd.Timestamp("2015-07-01")


def fetch_article(client, article, start, end, refresh=False):
    start, end = max(pd.Timestamp(start), FIRST_DAY), pd.Timestamp(end)
    url = URL.format(article=quote(article, safe=""), start=f"{start:%Y%m%d}00", end=f"{end:%Y%m%d}00")
    key = f"wiki/{quote(article, safe='')}_{start:%Y%m%d}_{end:%Y%m%d}.json"
    data = client.get_json(url, key, refresh)
    if not data:
        return None
    df = pd.DataFrame(data["items"])[["timestamp", "views"]]
    df["date"] = pd.to_datetime(df["timestamp"].str[:8])
    # A day's count is published after the day ends.
    df["available_at"] = df["date"] + pd.Timedelta(days=1)
    return df.drop(columns="timestamp").assign(article=article)


def build_pageviews(client, universe, start, end, refresh=False):
    """universe: DataFrame with columns ticker, wiki_articles (';'-separated)."""
    frames, missing = [], []
    for row in universe.itertuples():
        if not isinstance(row.wiki_articles, str):
            missing.append((row.ticker, None))
            continue
        for article in filter(None, (a.strip() for a in row.wiki_articles.split(";"))):
            df = fetch_article(client, article, start, end, refresh)
            if df is None:
                missing.append((row.ticker, article))
                continue
            frames.append(df.assign(ticker=row.ticker))
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return out, missing
