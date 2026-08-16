#!/usr/bin/env python3
"""
CI/CD regression gate.

Loads the 40-doc corpus and 50-pair golden dataset, ingests the corpus into
a fresh vector store, runs every golden query through the REAL RAG pipeline
(real embedding, real ACL filtering, real LLM generation), scores each
response with Ragas (real LLM-judged metrics), and writes a timestamped
JSON report to ci/reports/.

Exit code is non-zero if the aggregate faithfulness score drops below the
configured gate (default 0.85), which is what a GitHub Actions step checks
to block the deploy.

Usage:
    python backend/ci/run_regression.py
    python backend/ci/run_regression.py --gate 0.85 --limit 10   # quick smoke run
"""
from __future__ import annotations

import ssl
import certifi

# --- START SSL PATCH ---
# Force Python to use certifi instead of the corrupted Windows Certificate Store
def custom_ssl_context(*args, **kwargs):
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.load_verify_locations(certifi.where())
    return ctx

ssl.create_default_context = custom_ssl_context
# --- END SSL PATCH ---


import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # backend/ on path

from app.config import get_settings
from app.eval.failure_classifier import classify
from app.eval.ragas_eval import EvalSample, RagasEvaluator
from app.rag.pipeline import RAGPipeline
from app.rag.vector_store import Document, get_vector_store, new_id

CI_DIR = Path(__file__).parent


def load_fixtures() -> tuple[list[dict], list[dict]]:
    corpus = json.loads((CI_DIR / "corpus.json").read_text())
    golden = json.loads((CI_DIR / "golden_dataset.json").read_text())
    return corpus, golden


def ingest_corpus(store, corpus: list[dict]) -> None:
    docs = [Document(id=new_id(), text=d["text"], acl_role=d["acl_role"], source=d["source"]) for d in corpus]
    store.add_documents(docs)


async def run(gate: float, limit: int | None) -> dict:
    settings = get_settings()
    store = get_vector_store(settings)
    store.reset()

    corpus, golden = load_fixtures()
    ingest_corpus(store, corpus)
    print(f"Ingested {len(corpus)} documents into {settings.vector_store} ({store.count()} total).")

    samples = golden[:limit] if limit else golden
    pipeline = RAGPipeline(settings, store=store)
    evaluator = RagasEvaluator(settings)

    per_sample = []
    for i, item in enumerate(samples, 1):
        print(f"[{i}/{len(samples)}] role={item['role']!r} query={item['query']!r}")
        trace = pipeline.run(item["query"], item["role"])

        eval_sample = EvalSample(
            query=trace.query, answer=trace.answer, contexts=trace.context_used,
            ground_truth=item.get("ground_truth"),
        )
        result = await evaluator.evaluate_sample(eval_sample)
        report = classify(trace, faithfulness=result.faithfulness, context_precision=result.context_precision, context_recall=result.context_recall)

        per_sample.append({
            "query": item["query"],
            "role": item["role"],
            "expected_source": item.get("expected_source"),
            "answer": trace.answer,
            "retrieved_sources": [c.source for c in trace.retrieved_chunks],
            "faithfulness": result.faithfulness,
            "answer_relevancy": result.answer_relevancy,
            "context_precision": result.context_precision,
            "context_recall": result.context_recall,
            "root_cause": report.root_cause.value,
        })

    n = len(per_sample)
    avg_faithfulness = sum(s["faithfulness"] for s in per_sample) / n
    avg_relevancy = sum(s["answer_relevancy"] for s in per_sample) / n
    avg_precision = sum(s["context_precision"] for s in per_sample) / n
    passed = avg_faithfulness >= gate

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "llm_provider": settings.llm_provider,
        "vector_store": settings.vector_store,
        "total_samples": n,
        "avg_faithfulness": round(avg_faithfulness, 4),
        "avg_answer_relevancy": round(avg_relevancy, 4),
        "avg_context_precision": round(avg_precision, 4),
        "faithfulness_gate": gate,
        "passed": passed,
        "per_sample": per_sample,
    }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate", type=float, default=None, help="Faithfulness gate (default: settings.faithfulness_gate, 0.85)")
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N golden samples (smoke test)")
    args = parser.parse_args()

    settings = get_settings()
    gate = args.gate if args.gate is not None else settings.faithfulness_gate

    summary = asyncio.run(run(gate, args.limit))

    reports_dir = CI_DIR / "reports"
    reports_dir.mkdir(exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_path = reports_dir / f"{ts}.json"
    report_path.write_text(json.dumps(summary, indent=2))

    print()
    print("=" * 60)
    print(f"Samples run:            {summary['total_samples']}")
    print(f"Avg faithfulness:       {summary['avg_faithfulness']:.4f}  (gate: {gate:.2f})")
    print(f"Avg answer relevancy:   {summary['avg_answer_relevancy']:.4f}")
    print(f"Avg context precision:  {summary['avg_context_precision']:.4f}")
    print(f"Report written to:      {report_path}")
    print("=" * 60)

    if summary["passed"]:
        print("PASS — faithfulness gate cleared.")
        return 0
    else:
        print(f"FAIL — avg faithfulness {summary['avg_faithfulness']:.4f} is below the {gate:.2f} gate. Blocking deploy.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
