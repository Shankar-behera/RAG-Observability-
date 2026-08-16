from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/regression", tags=["regression"])

REPORTS_DIR = Path(__file__).resolve().parents[3] / "ci" / "reports"


@router.get("/latest")
def latest_report() -> dict:
    if not REPORTS_DIR.exists():
        raise HTTPException(status_code=404, detail="No regression reports found yet. Run backend/ci/run_regression.py.")
    reports = sorted(REPORTS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not reports:
        raise HTTPException(status_code=404, detail="No regression reports found yet. Run backend/ci/run_regression.py.")
    return json.loads(reports[0].read_text())


@router.get("/history")
def report_history(limit: int = 20) -> list[dict]:
    if not REPORTS_DIR.exists():
        return []
    reports = sorted(REPORTS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]
    out = []
    for r in reports:
        data = json.loads(r.read_text())
        out.append({
            "timestamp": r.stem,
            "avg_faithfulness": data.get("avg_faithfulness"),
            "avg_answer_relevancy": data.get("avg_answer_relevancy"),
            "avg_context_precision": data.get("avg_context_precision"),
            "passed": data.get("passed"),
            "total_samples": data.get("total_samples"),
        })
    return out
