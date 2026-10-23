from __future__ import annotations

import json
from json import JSONDecodeError
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from app.api.security import require_api_access

router = APIRouter(
    prefix="/regression",
    tags=["regression"],
    dependencies=[Depends(require_api_access)],
    responses={
        401: {"description": "Missing or invalid API access key."},
        503: {"description": "API access is not configured."},
    },
)

REPORTS_DIR = Path(__file__).resolve().parents[2] / "ci" / "reports"

NO_REPORTS_MESSAGE = (
    "No regression reports found yet. Run backend/ci/run_regression.py."
)
INVALID_REPORT_MESSAGE = "Regression report could not be read."


def _reports_directory_exists() -> bool:
    """Check report-directory availability without leaking filesystem details."""
    try:
        return REPORTS_DIR.exists()
    except OSError:
        raise HTTPException(
            status_code=500,
            detail=INVALID_REPORT_MESSAGE,
        ) from None


def _read_report(path: Path) -> dict[str, Any]:
    """Read a report without exposing filesystem or parser details."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, JSONDecodeError):
        raise HTTPException(
            status_code=500,
            detail=INVALID_REPORT_MESSAGE,
        ) from None

    if not isinstance(data, dict):
        raise HTTPException(
            status_code=500,
            detail=INVALID_REPORT_MESSAGE,
        )

    return data


def _reports_newest_first() -> list[Path]:
    """List reports by modification time, newest first."""
    try:
        return sorted(
            REPORTS_DIR.glob("*.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
    except OSError:
        raise HTTPException(
            status_code=500,
            detail=INVALID_REPORT_MESSAGE,
        ) from None


@router.get("/latest")
def latest_report() -> dict[str, Any]:
    if not _reports_directory_exists():
        raise HTTPException(status_code=404, detail=NO_REPORTS_MESSAGE)

    reports = _reports_newest_first()
    if not reports:
        raise HTTPException(status_code=404, detail=NO_REPORTS_MESSAGE)

    return _read_report(reports[0])


@router.get("/history")
def report_history(
    limit: int = Query(
        default=20,
        ge=1,
        le=100,
        description="Maximum number of recent regression reports to return (1–100).",
    ),
) -> list[dict[str, Any]]:
    if not _reports_directory_exists():
        return []

    reports = _reports_newest_first()[:limit]
    out = []

    for report_path in reports:
        data = _read_report(report_path)
        out.append(
            {
                "timestamp": report_path.stem,
                "avg_faithfulness": data.get("avg_faithfulness"),
                "avg_answer_relevancy": data.get("avg_answer_relevancy"),
                "avg_context_precision": data.get("avg_context_precision"),
                "passed": data.get("passed"),
                "total_samples": data.get("total_samples"),
            }
        )

    return out
