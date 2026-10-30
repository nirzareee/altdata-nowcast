"""Revenue guidance extracted from earnings press releases by an LLM.

Three stages, each checkable on its own:
1. Filter: keep only passages that look like forward-looking statements with
   numbers. Most of a release is financial tables; the outlook is a few paragraphs.
2. Extract: ask the model for structured guidance, each item with the exact
   sentence it came from. Any OpenAI-compatible server works (vLLM, Ollama, hosted).
3. Verify: the quote must appear in the passage and each number in the quote.
   An item that fails is kept but flagged, never silently trusted.

Results are cached per release and model, so a job that hits a time limit resumes.
"""
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd
import requests

CACHE = Path("data/raw/guidance")
PROMPT_VERSION = "v2"  # part of the cache path: changing the prompt reruns every release
MAX_PASSAGE_CHARS = 8000

FORWARD = re.compile(
    r"\b(outlook|guidance|expect\w*|anticipat\w*|forecast\w*|project\w*|reaffirm\w*|"
    r"rais\w+ (its|our|the)|lower\w* (its|our|the)|full[- ]year|fiscal (year )?20\d\d|"
    r"for the (first|second|third|fourth) quarter)\b",
    re.IGNORECASE,
)
NUMBER = re.compile(r"\d")

SYSTEM = """You extract revenue guidance from company earnings press releases.
Return JSON only: {"guidance": [ ... ]}. Use an empty list if the text gives no revenue guidance.

Guidance is what management EXPECTS for a FUTURE period ("expects", "outlook", "guidance",
"anticipates", a guidance table). Never extract results already reported for a past period
("revenue grew 21% to $864 million" is a result, not guidance).

Each item:
- "metric": "revenue" (net sales, total revenues, organic or comparable sales growth)
- "scope": "total" for total company revenue, "segment" for a business line, region or product
- "basis": "reported", "organic", "comparable", "constant_currency" or "other"
- "period_type": "quarter" or "year"
- "fiscal_year": integer, e.g. 2024
- "fiscal_quarter": 1-4 for quarterly guidance, null for annual
- "kind": "level" for an amount or "growth" for a percent change
- "low", "high": the numbers exactly as printed, never converted ($2.65 billion -> 2.65; a single value: low = high)
- "scale": for levels, "units", "thousands", "millions" or "billions"; null for growth
- "quote": the sentence or table row the numbers come from, copied character for character

Example. Text: "For fiscal 2024 we expect net sales of $2.1 to $2.2 billion, with Americas sales up 3% to 4%."
{"guidance": [
 {"metric": "revenue", "scope": "total", "basis": "reported", "period_type": "year", "fiscal_year": 2024,
  "fiscal_quarter": null, "kind": "level", "low": 2.1, "high": 2.2, "scale": "billions",
  "quote": "For fiscal 2024 we expect net sales of $2.1 to $2.2 billion, with Americas sales up 3% to 4%."},
 {"metric": "revenue", "scope": "segment", "basis": "reported", "period_type": "year", "fiscal_year": 2024,
  "fiscal_quarter": null, "kind": "growth", "low": 3, "high": 4, "scale": null,
  "quote": "For fiscal 2024 we expect net sales of $2.1 to $2.2 billion, with Americas sales up 3% to 4%."}
]}

Only revenue. Ignore EPS, margins, EBITDA, capex and share counts."""


def outlook_passages(text, context=1):
    """Lines that look forward-looking and contain a number, with `context` neighbors."""
    lines = [l for l in text.splitlines() if l.strip()]
    keep = set()
    for i, line in enumerate(lines):
        if FORWARD.search(line) and NUMBER.search(line):
            keep.update(range(max(0, i - context), min(len(lines), i + context + 1)))
    passage = "\n".join(lines[i] for i in sorted(keep))
    return passage[:MAX_PASSAGE_CHARS]


def _numbers(s):
    return {float(x.replace(",", "")) for x in re.findall(r"\d[\d,]*\.?\d*", s)}


def _norm(s):
    """Letters, digits, '.', '%' only. HTML tables come out of text extraction with cells
    glued together or split across lines, so spacing and '$' or dash characters can't be trusted."""
    return re.sub(r"[^a-z0-9.%]", "", s.lower())


SCALES = (1, 1e3, 1e6, 1e9)


def _in_quote(value, nums):
    """Value appears in the quote as printed, or converted to units from thousands/millions/billions."""
    v = abs(float(value))
    return any(abs(v / s - n) < 1e-6 * max(1, n) for s in SCALES for n in nums)


def _quote_in(passage, quote, min_share=0.95):
    """At least 95% of the quote's characters appear in the passage, in order. Exact matching
    fails on tables, where extraction glues in footnote markers the model rightly drops."""
    q, p = _norm(quote), _norm(passage)
    if not q:
        return False
    if q in p:
        return True
    matched = sum(b.size for b in SequenceMatcher(None, p, q, autojunk=False).get_matching_blocks())
    return matched / len(q) >= min_share


def verify(item, passage):
    """Quote (nearly) appears in the passage, and low/high appear as numbers in both the quote
    and the passage, as printed or scaled to units."""
    quote = item.get("quote") or ""
    if not _quote_in(passage, quote):
        return False
    in_quote, in_passage = _numbers(quote), _numbers(passage)
    values = [v for v in (item.get("low"), item.get("high")) if v is not None]
    try:
        return bool(values) and all(_in_quote(v, in_quote) and _in_quote(v, in_passage) for v in values)
    except (TypeError, ValueError):
        return False


def call_llm(base_url, model, passage, api_key=None, timeout=300):
    r = requests.post(
        f"{base_url.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {api_key or 'none'}"},
        json={
            "model": model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": passage}],
        },
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def parse_items(content):
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, re.DOTALL)  # model wrapped the JSON in prose
        data = json.loads(match.group(0)) if match else {}
    items = data.get("guidance", []) if isinstance(data, dict) else []
    return [i for i in items if isinstance(i, dict) and i.get("metric") == "revenue"
            and i.get("scope", "total") == "total"]


def model_slug(model):
    return re.sub(r"[^A-Za-z0-9.]+", "_", model).strip("_")


def extract_release(row, base_url, model, api_key=None, call=call_llm):
    """Guidance items for one release (cached). Returns a list of dicts."""
    path = CACHE / model_slug(model) / PROMPT_VERSION / f"{row['accession']}.json"
    if path.exists():
        return json.loads(path.read_text())
    passage = outlook_passages(Path(row["path"]).read_text())
    if not passage:
        items, error = [], None
    else:
        try:
            items, error = parse_items(call(base_url, model, passage, api_key)), None
        except Exception as e:  # not cached: retried on the next run
            return [{"error": str(e)}]
        for item in items:
            item["verified"] = verify(item, passage)
    result = [{**i, "accession": row["accession"], "ticker": row["ticker"]} for i in items]
    if not result:
        result = [{"accession": row["accession"], "ticker": row["ticker"], "none": True,
                   "passage_chars": len(passage)}]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result))
    return result


def build_guidance(releases, base_url, model, api_key=None, workers=16, call=call_llm, progress=None):
    rows = releases.to_dict("records")
    out, errors = [], 0
    with ThreadPoolExecutor(workers) as pool:
        futures = [pool.submit(extract_release, r, base_url, model, api_key, call) for r in rows]
        for n, f in enumerate(as_completed(futures), 1):
            items = f.result()
            if any("error" in i for i in items):
                errors += 1
            else:
                out.extend(items)
            if progress and n % 100 == 0:
                progress(n, len(rows), errors)
    df = pd.DataFrame(out)
    if df.empty:
        return df, errors
    df = df.merge(releases[["accession", "available_at"]], on="accession", how="left")
    return df, errors
