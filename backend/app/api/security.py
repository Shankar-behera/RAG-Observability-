from __future__ import annotations

import hmac

from fastapi import Header, HTTPException

from app.config import get_settings


def require_api_access(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> None:
    """Fail closed unless a configured service API key matches."""
    configured_key = get_settings().api_access_key

    if not configured_key or not configured_key.strip():
        raise HTTPException(
            status_code=503,
            detail="API access is not configured.",
        )

    configured_key = configured_key.strip()
    if x_api_key is None or not hmac.compare_digest(
        x_api_key.encode("utf-8"),
        configured_key.encode("utf-8"),
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key.",
        )
