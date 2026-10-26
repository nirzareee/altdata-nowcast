"""Company universe by a fixed rule: S&P 500 members in consumer-facing GICS sectors.

The constituents table on Wikipedia gives each firm's GICS sector, CIK and a
link to its Wikipedia article, so the page mapping comes from the source
rather than from hand-picked titles.
"""
from io import StringIO
from urllib.parse import unquote

import pandas as pd

SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
SECTORS = ["Consumer Discretionary", "Consumer Staples", "Communication Services"]


def _text(cell):
    return cell[0] if isinstance(cell, tuple) else cell


WIKI_PREFIXES = ("https://en.wikipedia.org/wiki/", "//en.wikipedia.org/wiki/", "/wiki/", "./")


def _article(cell):
    """Article title from a link; Wikipedia serves '/wiki/Title' or, in newer HTML, './Title'."""
    href = cell[1] if isinstance(cell, tuple) else None
    if not href:
        return None
    for prefix in WIKI_PREFIXES:
        if href.startswith(prefix):
            title = unquote(href.removeprefix(prefix).split("#")[0])
            return title if title and ":" not in title else None
    return None


def parse_constituents(html, sectors=SECTORS):
    tables = pd.read_html(StringIO(html), extract_links="body")
    table = next(t for t in tables if {"Symbol", "Security", "GICS Sector"} <= set(t.columns))
    df = pd.DataFrame({
        "ticker": table["Symbol"].map(_text).str.replace(".", "-", regex=False),
        "company": table["Security"].map(_text),
        "sector": table["GICS Sector"].map(_text),
        "sub_industry": table["GICS Sub-Industry"].map(_text),
        "cik": table["CIK"].map(_text).astype(int),
        "wiki_articles": table["Security"].map(_article),
    })
    df = df[df["sector"].isin(sectors)]
    # Dual share classes (GOOGL/GOOG, FOXA/FOX) are one company.
    return df.drop_duplicates("cik").sort_values(["sector", "ticker"]).reset_index(drop=True)


def build_universe(client, refresh=False):
    html = client.get_text(SP500_URL, "universe/sp500.html", refresh)
    return parse_constituents(html)
