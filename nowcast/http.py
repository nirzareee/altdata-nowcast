import json
import os
import time
from pathlib import Path

import requests

CACHE = Path(os.environ.get("NOWCAST_CACHE", "data/raw"))


class Client:
    """Rate-limited JSON client with an on-disk cache.

    SEC and Wikimedia both reject requests without a descriptive User-Agent.
    """

    def __init__(self, user_agent=None, min_interval=0.15, retries=3):
        ua = user_agent or os.environ.get("NOWCAST_USER_AGENT")
        if not ua:
            raise RuntimeError("Set NOWCAST_USER_AGENT, e.g. 'Jane Doe jane@school.edu'")
        self.session = requests.Session()
        self.session.headers["User-Agent"] = ua
        self.min_interval = min_interval
        self.retries = retries
        self._last = 0.0

    def get_json(self, url, cache_key=None, refresh=False):
        path = CACHE / cache_key if cache_key else None
        if path and path.exists() and not refresh:
            return json.loads(path.read_text())
        r = self._get(url)
        if r is None:
            return None
        data = r.json()
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data))
        return data

    def get_text(self, url, cache_key=None, refresh=False):
        path = CACHE / cache_key if cache_key else None
        if path and path.exists() and not refresh:
            return path.read_text()
        r = self._get(url)
        if r is None:
            return None
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(r.text)
        return r.text

    def _get(self, url):
        for attempt in range(self.retries):
            wait = self.min_interval - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            r = self.session.get(url, timeout=30)
            self._last = time.monotonic()
            if r.status_code == 404:
                return None
            if r.status_code in (429, 500, 502, 503) and attempt < self.retries - 1:
                time.sleep(2 ** attempt)
                continue
            r.raise_for_status()
            return r
