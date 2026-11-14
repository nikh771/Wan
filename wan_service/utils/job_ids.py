"""Job id validation for API routes."""

from __future__ import annotations

import re

from fastapi import HTTPException

# Matches job_manager.create_job: <engine>_<16 hex chars> (wan | ltx | cogvideox)
JOB_ID_PATTERN = re.compile(
    r"^(?:wan|ltx|cogvideox)_[0-9a-f]{16}$",
    re.IGNORECASE,
)

_PLACEHOLDER_IDS = frozenset(
    {
        "your_job_id",
        "job_id",
        "wan_xxxxxxxxxxxxxxxx",
    }
)


def require_job_id(job_id: str) -> str:
    """Return normalized job id or raise HTTPException (400/404)."""
    normalized = (job_id or "").strip()
    if not normalized:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "job_id is required",
                "revoke_polling": True,
            },
        )
    if normalized.lower() in _PLACEHOLDER_IDS:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Replace the placeholder job_id with the id returned from POST /jobs",
                "job_id": normalized,
                "revoke_polling": True,
            },
        )
    if not JOB_ID_PATTERN.match(normalized):
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Invalid job_id format; expected {wan|ltx|cogvideox}_<16 hex characters>",
                "job_id": normalized,
                "revoke_polling": True,
            },
        )
    return normalized.lower()
