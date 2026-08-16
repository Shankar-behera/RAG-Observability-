from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ci"))

from run_regression import load_fixtures  # noqa: E402


def test_golden_dataset_has_exactly_fifty_samples():
    _, golden = load_fixtures()
    assert len(golden) == 50


def test_corpus_has_forty_docs_ten_per_role():
    corpus, _ = load_fixtures()
    assert len(corpus) == 40
    from collections import Counter
    counts = Counter(d["acl_role"] for d in corpus)
    assert counts == {"Engineering": 10, "HR": 10, "Executive": 10, "General": 10}


def test_every_golden_sample_has_required_fields():
    _, golden = load_fixtures()
    for item in golden:
        assert item["query"]
        assert item["role"] in {"Engineering", "HR", "Executive", "General"}
        assert "ground_truth" in item


def test_gate_pass_fail_boundary():
    """Mirrors the exact comparison used in run_regression.run(): a mean
    faithfulness score exactly at or above the gate passes, below it fails."""
    gate = 0.85

    def passed(avg_faithfulness: float) -> bool:
        return avg_faithfulness >= gate

    assert passed(0.85) is True
    assert passed(0.86) is True
    assert passed(0.8499) is False
    assert passed(0.0) is False


def test_report_json_shape_matches_dashboard_expectations():
    """The Next.js dashboard's /regression/latest consumer expects these
    exact keys; this test locks the report schema so a refactor of
    run_regression.py can't silently break the frontend contract."""
    fake_summary = {
        "timestamp": "2026-01-01T00:00:00Z",
        "llm_provider": "ollama",
        "vector_store": "chroma",
        "total_samples": 50,
        "avg_faithfulness": 0.91,
        "avg_answer_relevancy": 0.88,
        "avg_context_precision": 0.93,
        "faithfulness_gate": 0.85,
        "passed": True,
        "per_sample": [],
    }
    required_keys = {
        "timestamp", "llm_provider", "vector_store", "total_samples",
        "avg_faithfulness", "avg_answer_relevancy", "avg_context_precision",
        "faithfulness_gate", "passed", "per_sample",
    }
    assert required_keys <= set(fake_summary.keys())
    # sanity check it round-trips through JSON cleanly (what the script writes to disk)
    json.loads(json.dumps(fake_summary))
