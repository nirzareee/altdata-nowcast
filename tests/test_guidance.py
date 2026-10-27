import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pandas as pd
import pytest

from nowcast import guidance, labels
from nowcast.guidance import build_guidance, call_llm, outlook_passages, parse_items, verify

RELEASE = """Coca-Cola Reports First Quarter 2024 Results
Net revenues grew 3% to $11.3 billion.
CONSOLIDATED STATEMENTS OF INCOME
Net operating revenues 11,300 10,980
Cost of goods sold 4,500 4,390
2024 Outlook
The company now expects to deliver organic revenue growth of 8% to 9% for full-year 2024.
For the second quarter, the company expects net revenues of $11.9 billion to $12.1 billion.
Forward-looking statements are subject to risks."""

GOOD = {"guidance": [
    {"metric": "revenue", "basis": "organic", "period_type": "year", "fiscal_year": 2024, "fiscal_quarter": None,
     "kind": "growth", "low": 8, "high": 9, "scale": None,
     "quote": "The company now expects to deliver organic revenue growth of 8% to 9% for full-year 2024."},
    {"metric": "revenue", "basis": "reported", "period_type": "quarter", "fiscal_year": 2024, "fiscal_quarter": 2,
     "kind": "level", "low": 11.9, "high": 12.1, "scale": "billions",
     "quote": "For the second quarter, the company expects net revenues of $11.9 billion to $12.1 billion."},
    {"metric": "eps", "low": 2.8, "high": 2.85, "quote": "x"},
]}


def test_filter_keeps_outlook_drops_tables():
    p = outlook_passages(RELEASE, context=0)
    assert "organic revenue growth of 8% to 9%" in p
    assert "Cost of goods sold" not in p


def test_verify_catches_invented_numbers_and_quotes():
    item = GOOD["guidance"][0]
    assert verify(item, RELEASE)
    assert not verify({**item, "high": 10}, RELEASE)  # number not in the quote
    assert not verify({**item, "quote": "Organic growth will be 8% to 9%."}, RELEASE)  # paraphrase
    assert verify(GOOD["guidance"][1], RELEASE)


def test_parse_items_keeps_revenue_and_tolerates_prose():
    assert len(parse_items(json.dumps(GOOD))) == 2
    assert len(parse_items("Here you go:\n" + json.dumps(GOOD) + "\nThanks")) == 2
    assert parse_items("not json") == []


@pytest.fixture
def releases(tmp_path, monkeypatch):
    monkeypatch.setattr(guidance, "CACHE", tmp_path / "cache")
    texts = {"A-1": RELEASE, "A-2": "Quarterly results. Revenue was $5 million.\nThank you."}
    rows = []
    for acc, text in texts.items():
        p = tmp_path / f"{acc}.txt"
        p.write_text(text)
        rows.append({"accession": acc, "ticker": "KO", "path": str(p), "filed": "2024-04-30",
                     "available_at": pd.Timestamp("2024-05-01")})
    return pd.DataFrame(rows)


def test_build_guidance_caches_and_skips_releases_without_outlook(releases):
    calls = []

    def fake(base_url, model, passage, api_key=None):
        calls.append(passage)
        return json.dumps(GOOD)

    df, errors = build_guidance(releases, "http://x", "test/model", call=fake, workers=2)
    assert errors == 0 and len(calls) == 1  # A-2 has no forward-looking passage, so no LLM call
    items = df[df["none"] != True]  # noqa: E712
    assert len(items) == 2 and items["verified"].all()
    assert (items["available_at"] == pd.Timestamp("2024-05-01")).all()
    build_guidance(releases, "http://x", "test/model", call=fake)
    assert len(calls) == 1  # second run served from cache


def test_server_errors_are_not_cached(releases):
    def failing(*a, **k):
        raise RuntimeError("503")

    df, errors = build_guidance(releases, "http://x", "m", call=failing)
    assert errors == 1
    df, errors = build_guidance(releases, "http://x", "m", call=lambda *a, **k: json.dumps(GOOD))
    assert errors == 0 and (df["none"] != True).sum() == 2  # noqa: E712


def test_call_llm_speaks_openai_protocol():
    seen = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            seen["path"] = self.path
            seen["body"] = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            body = json.dumps({"choices": [{"message": {"content": json.dumps(GOOD)}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        out = call_llm(f"http://127.0.0.1:{server.server_port}/v1", "m", "passage")
    finally:
        server.shutdown()
    assert seen["path"] == "/v1/chat/completions"
    assert seen["body"]["temperature"] == 0 and seen["body"]["response_format"] == {"type": "json_object"}
    assert json.loads(out) == GOOD


def test_label_sample_and_score(releases, tmp_path, monkeypatch):
    monkeypatch.setattr(labels, "LABELS", tmp_path / "labels")
    path = labels.make_sample(releases, n=2)
    sheet = pd.read_csv(path)
    assert set(sheet["accession"]) == {"A-1", "A-2"}

    gold = pd.DataFrame([
        {"accession": "A-1", "has_guidance": "y", "period_type": "year", "fiscal_year": 2024,
         "fiscal_quarter": "", "kind": "growth", "low": 8, "high": 9},
        {"accession": "A-1", "has_guidance": "y", "period_type": "quarter", "fiscal_year": 2024,
         "fiscal_quarter": 2, "kind": "level", "low": 11.9, "high": 12.0},  # model says 12.1
        {"accession": "A-2", "has_guidance": "n"},
    ])
    preds, _ = build_guidance(releases, "http://x", "m", call=lambda *a, **k: json.dumps(GOOD))
    r = labels.score(gold, preds)
    assert r["detection precision"] == 1 and r["detection recall"] == 1
    assert r["item recall"] == 1
    assert r["value accuracy (matched items)"] == 0.5
