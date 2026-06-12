from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from cardionet.io.common import normalize_patient_id, normalize_slice_type

REQUIRED_DATA_INDEX_COLUMNS = ("ID", "Region", "Slice")


@dataclass(frozen=True, slots=True)
class SelectedSlice:
    """Clinician-selected AHA slice metadata for one patient."""

    patient_id: str
    slice_index: int
    slice_type: str


SelectedSlicesByPatient = dict[str, list[SelectedSlice]]


def append_selected_slice(
    selected: SelectedSlicesByPatient,
    *,
    patient_id: str,
    slice_index: int,
    slice_type: str,
) -> None:
    """Append one clinician-selected slice while preserving input order."""
    selected.setdefault(patient_id, []).append(
        SelectedSlice(
            patient_id=patient_id,
            slice_index=int(slice_index),
            slice_type=normalize_slice_type(slice_type),
        )
    )


def parse_selected_slices(selection_json: str | None) -> SelectedSlicesByPatient:
    """Parse selected slices from a JSON mapping keyed by patient ID."""
    raw = json.loads(selection_json or "{}")
    selected: SelectedSlicesByPatient = {}
    for patient_key, payload in raw.items():
        patient_id = normalize_patient_id(patient_key)
        payloads = payload if isinstance(payload, list) else [payload]
        for item in payloads:
            if not isinstance(item, dict):
                raise ValueError(
                    f"Selection for {patient_key!r} must be a JSON object or list of objects."
                )
            append_selected_slice(
                selected,
                patient_id=patient_id,
                slice_index=int(item.get("slice_index", item.get("slice"))),
                slice_type=str(item.get("slice_type", item.get("region"))),
            )
    return selected


def resolve_data_index_path(input_root: Path, data_index_path: str | None) -> Path | None:
    """Resolve an optional index path relative to input_root."""
    if not data_index_path:
        return None
    path = Path(str(data_index_path).strip().strip('"').strip("'"))
    if not path.is_absolute():
        path = input_root / path
    return path.resolve()


def read_data_index_file(path: Path) -> pd.DataFrame:
    """Read an ID/Region/Slice index from xlsx, csv, or delimited txt."""
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".txt":
        return pd.read_csv(path, sep=None, engine="python")
    raise ValueError(
        f"Unsupported data index extension {path.suffix!r}; expected .xlsx, .csv, or .txt."
    )


def scalar_slice_index(value: object, *, row_number: int) -> int:
    """Parse one scalar slice index and reject pre-coded array values."""
    if pd.isna(value):
        raise ValueError(f"Data index row {row_number}: Slice is blank.")
    text = str(value).strip()
    if any(marker in text for marker in ("[", "]", "{", "}", "(", ")", ",")):
        raise ValueError(
            f"Data index row {row_number}: Slice must be one scalar value, got {value!r}."
        )
    return int(float(text))


def load_data_index_frame(path: Path) -> pd.DataFrame:
    """Load one-slice-per-line index rows into a patient-indexed dataframe."""
    if not path.exists():
        raise FileNotFoundError(f"Data index file not found: {path}")

    raw = read_data_index_file(path)
    missing = [column for column in REQUIRED_DATA_INDEX_COLUMNS if column not in raw.columns]
    if missing:
        raise ValueError(
            f"Data index {path} is missing required column(s): {missing}. "
            f"Expected headers: {list(REQUIRED_DATA_INDEX_COLUMNS)}"
        )

    rows: list[dict[str, int | str]] = []
    for dataframe_index, row in raw.iterrows():
        excel_row = int(dataframe_index) + 2
        rows.append(
            {
                "patient_id": normalize_patient_id(row["ID"]),
                "slice_index": scalar_slice_index(row["Slice"], row_number=excel_row),
                "slice_type": normalize_slice_type(str(row["Region"])),
                "row_order": int(dataframe_index),
            }
        )

    if not rows:
        raise ValueError(f"Data index has no selected slice rows: {path}")

    normalized = pd.DataFrame(rows).sort_values("row_order", kind="stable")
    duplicates = normalized.duplicated(["patient_id", "slice_index", "slice_type"])
    if bool(duplicates.any()):
        duplicate_rows = normalized.loc[duplicates, ["patient_id", "slice_index", "slice_type"]]
        raise ValueError(
            "Data index contains duplicate patient/slice/type row(s): "
            + repr(duplicate_rows.to_dict(orient="records"))
        )

    grouped = normalized.groupby("patient_id", sort=False).agg(
        slice_indices=("slice_index", list),
        slice_types=("slice_type", list),
    )
    grouped.index.name = "ID"
    return grouped


def selected_slices_from_index_frame(index_frame: pd.DataFrame) -> SelectedSlicesByPatient:
    """Convert a grouped index dataframe into ordered SelectedSlice objects."""
    selected: SelectedSlicesByPatient = {}
    for patient_id, row in index_frame.iterrows():
        slice_indices = list(row["slice_indices"])
        slice_types = list(row["slice_types"])
        if len(slice_indices) != len(slice_types):
            raise ValueError(
                f"Data index patient {patient_id} has mismatched slice/type counts: "
                f"{len(slice_indices)} vs {len(slice_types)}"
            )
        for slice_index, slice_type in zip(slice_indices, slice_types):
            append_selected_slice(
                selected,
                patient_id=str(patient_id),
                slice_index=int(slice_index),
                slice_type=str(slice_type),
            )
    return selected


def selected_slice_iter(
    selected_slices: SelectedSlicesByPatient,
) -> list[tuple[str, SelectedSlice]]:
    """Flatten patient-indexed selected slices while preserving input order."""
    return [
        (patient_id, selected_slice)
        for patient_id, patient_slices in selected_slices.items()
        for selected_slice in patient_slices
    ]
