from __future__ import annotations

import json
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from cardionet.pipelines.cluster_manifest import PipelineCase


@dataclass(frozen=True, slots=True)
class StageStatus:
    """JSON-serializable status record for one stage/case run."""

    stage: str
    patient_id: str | None
    case_index: int | None
    status: str
    message: str
    started_at: str
    finished_at: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def status_path(status_root: Path, *, stage: str, case: PipelineCase | None) -> Path:
    """Return a stable status path for one stage/case."""
    if case is None:
        filename = f"{stage}.json"
    else:
        filename = f"{case.case_index:06d}_{case.patient_id}.json"
    return status_root / stage / filename


def write_status(
    status_root: Path,
    *,
    stage: str,
    case: PipelineCase | None,
    status: str,
    message: str = "",
    started_at: str | None = None,
) -> Path:
    """Write a stage status JSON record."""
    started = started_at or utc_now()
    record = StageStatus(
        stage=stage,
        patient_id=case.patient_id if case is not None else None,
        case_index=case.case_index if case is not None else None,
        status=status,
        message=message,
        started_at=started,
        finished_at=utc_now(),
    )
    path = status_path(status_root, stage=stage, case=case)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(record), indent=2), encoding="utf-8")
    return path


def format_exception(exc: BaseException) -> str:
    """Return a compact traceback string for status files."""
    return "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))


def collect_status(status_root: Path) -> dict[str, object]:
    """Summarize all status JSON records below a status root."""
    records = []
    for path in sorted(status_root.glob("*/*.json")):
        records.append(json.loads(path.read_text(encoding="utf-8")))

    counts: dict[str, dict[str, int]] = {}
    for record in records:
        stage = str(record["stage"])
        state = str(record["status"])
        counts.setdefault(stage, {})
        counts[stage][state] = counts[stage].get(state, 0) + 1

    failures = [
        record
        for record in records
        if str(record.get("status")) == "failed"
    ]
    return {
        "status_root": str(status_root),
        "record_count": len(records),
        "counts": counts,
        "failures": failures,
    }
