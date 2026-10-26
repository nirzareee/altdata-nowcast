"""Earnings press releases from SEC 8-K filings.

Companies furnish their quarterly results under 8-K Item 2.02 ("Results of
Operations"), with the press release attached as Exhibit 99. The filing's
acceptance timestamp is when the text became public. EDGAR holds filings accepted
after 5:30 pm until the next business day, so a release counts as known from the
start of the day after acceptance; nowcasts come weeks later, so this costs nothing.
"""
import re
from io import StringIO
from pathlib import Path

import lxml.html
import pandas as pd

SUBMISSIONS = "https://data.sec.gov/submissions/{name}"
ARCHIVE = "https://www.sec.gov/Archives/edgar/data/{cik}/{folder}/{accession}-index.htm"
SEC = "https://www.sec.gov"
TEXT_DIR = Path("data/raw/releases")


def _filings_table(block):
    keys = ["accessionNumber", "filingDate", "acceptanceDateTime", "form", "items"]
    return pd.DataFrame({k: block.get(k, []) for k in keys})


def earnings_filings(client, cik, since="2015-06-01"):
    """All 8-K filings with Item 2.02 since `since`, newest pages included."""
    data = client.get_json(SUBMISSIONS.format(name=f"CIK{cik:010d}.json"), f"sec/submissions_{cik:010d}.json")
    tables = [_filings_table(data["filings"]["recent"])]
    for page in data["filings"].get("files", []):
        if page.get("filingTo", "9999") < since:
            continue
        tables.append(_filings_table(client.get_json(SUBMISSIONS.format(name=page["name"]), f"sec/{page['name']}")))
    df = pd.concat(tables, ignore_index=True)
    df = df[df["form"].isin(["8-K", "8-K/A"]) & df["items"].fillna("").str.contains("2.02", regex=False)]
    df = df[df["filingDate"] >= since].drop_duplicates("accessionNumber")
    df["accepted"] = pd.to_datetime(df["acceptanceDateTime"], utc=True).dt.tz_convert(None)
    return df.sort_values("accepted").reset_index(drop=True)


def exhibit_url(client, cik, accession):
    """URL of the press-release exhibit (EX-99.1 preferred, then any EX-99)."""
    url = ARCHIVE.format(cik=cik, folder=accession.replace("-", ""), accession=accession)
    html = client.get_text(url, f"sec/index/{accession}.htm")
    if not html:
        return None
    for t in pd.read_html(StringIO(html), extract_links="body"):
        if not {"Type", "Document"} <= set(t.columns):
            continue
        types = t["Type"].map(lambda c: (c[0] if isinstance(c, tuple) else str(c)) or "")
        for wanted in ("EX-99.1", "EX-99"):
            hits = t[types.str.upper().str.startswith(wanted)]
            for cell in hits["Document"]:
                href = cell[1] if isinstance(cell, tuple) else None
                if href:
                    return SEC + href.replace("/ix?doc=", "")
    return None


XML_DECLARATION = re.compile(r"^\s*<\?xml[^>]*\?>", re.IGNORECASE)


def html_to_text(raw):
    if not raw.lstrip().startswith("<"):
        return raw
    # XHTML exhibits open with <?xml ... encoding=...?>; the text is already decoded,
    # and lxml rejects a str that still declares an encoding.
    doc = lxml.html.fromstring(XML_DECLARATION.sub("", raw, count=1))
    for bad in doc.xpath("//script|//style"):
        bad.drop_tree()
    lines = (" ".join(line.split()) for line in doc.text_content().splitlines())
    return "\n".join(line for line in lines if line)


def build_releases(client, universe, since="2015-06-01"):
    rows, missing = [], []
    for firm in universe.itertuples():
        try:
            filings = earnings_filings(client, int(firm.cik), since)
        except Exception as e:  # one bad company shouldn't stop 99
            missing.append((firm.ticker, f"submissions: {e}"))
            continue
        for f in filings.itertuples():
            path = TEXT_DIR / firm.ticker / f"{f.accessionNumber}.txt"
            if not path.exists():
                url = exhibit_url(client, int(firm.cik), f.accessionNumber)
                raw = client.get_text(url, f"sec/exhibits/{f.accessionNumber}.htm") if url else None
                try:
                    text = html_to_text(raw) if raw else ""
                except Exception as e:  # malformed exhibit: log it, keep going
                    missing.append((firm.ticker, f"{f.accessionNumber}: {e}"))
                    continue
                if not text:
                    missing.append((firm.ticker, f.accessionNumber))
                    continue
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text)
            rows.append({"ticker": firm.ticker, "cik": int(firm.cik), "accession": f.accessionNumber,
                         "form": f.form, "filed": pd.Timestamp(f.filingDate),
                         "available_at": f.accepted.normalize() + pd.Timedelta(days=1),
                         "path": str(path), "chars": path.stat().st_size})
    return pd.DataFrame(rows), missing
