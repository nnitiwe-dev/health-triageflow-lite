"""Resilient file download — resumable + retrying, for model weights and binaries.

Survives network interruptions: a partial ``.part`` file is resumed with an HTTP Range
request on the next attempt, with exponential backoff between retries. Used for GGUF model
files and (on Windows) the llama.cpp binary. ``trust_env=False`` so a stray proxy env var
can't break a localhost-adjacent fetch.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

import httpx


def download_with_resume(url: str, dest: Path, *, label: str = "",
                         on_progress: Callable[[str], None] | None = None,
                         retries: int = 4, headers: dict | None = None) -> bool:
    """Download ``url`` to ``dest`` (atomically via a ``.part`` file). Returns True on success.

    Resumes from a partial file across retries; backs off on failure. Never raises — a failed
    download returns False (and leaves the ``.part`` so a later call can resume it)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    label = label or dest.name

    for attempt in range(retries + 1):
        existing = tmp.stat().st_size if tmp.exists() else 0
        req_headers = dict(headers or {})
        mode = "wb"
        if existing:
            req_headers["Range"] = f"bytes={existing}-"
            mode = "ab"
        try:
            with httpx.stream("GET", url, follow_redirects=True, timeout=None,
                              headers=req_headers, trust_env=False) as r:
                if r.status_code == 416:  # range beyond file → already complete
                    tmp.rename(dest)
                    return True
                if existing and r.status_code == 200:  # server ignored Range → restart clean
                    existing, mode = 0, "wb"
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0)) + existing
                done = existing
                last = done
                with open(tmp, mode) as f:
                    for chunk in r.iter_bytes(chunk_size=1 << 20):
                        f.write(chunk)
                        done += len(chunk)
                        if on_progress and total and done - last > (20 << 20):  # ~every 20 MB
                            last = done
                            on_progress(f"{label} — {100 * done / total:.0f}% "
                                        f"({done >> 20}/{total >> 20} MB)")
            tmp.rename(dest)
            if on_progress:
                on_progress(f"{label} — done")
            return True
        except Exception as e:
            if on_progress:
                on_progress(f"{label} — interrupted ({type(e).__name__}); "
                            f"retry {attempt + 1}/{retries}…")
            if attempt < retries:
                time.sleep(min(2 ** attempt, 15))
    return False
