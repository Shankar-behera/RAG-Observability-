#!/usr/bin/env python3
"""
Seeds a RUNNING backend instance (via its real /ingest HTTP endpoint) with
the 40-doc synthetic corpus used by the golden dataset. Useful for manually
exploring the /query and /debug/trace UI against realistic data without
running the full CI regression script.

Usage:
    python backend/scripts/seed_corpus.py [--api-url http://localhost:8000]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import httpx

CORPUS_PATH = Path(__file__).resolve().parents[1] / "ci" / "corpus.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-url", default="http://localhost:8000")
    args = parser.parse_args()

    corpus = json.loads(CORPUS_PATH.read_text())
    documents = [{"text": d["text"], "acl_role": d["acl_role"], "source": d["source"]} for d in corpus]

    resp = httpx.post(f"{args.api_url}/ingest", json={"documents": documents}, timeout=60)
    resp.raise_for_status()
    result = resp.json()
    print(f"Ingested {result['ingested']} documents. Total in store: {result['total_in_store']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
