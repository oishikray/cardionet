from __future__ import annotations

import argparse
import ast
import csv
import os
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile
from xml.etree import ElementTree as ET

from cardionet.config import load_cardionet_config

XLSX_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"

SEGMENTS_BY_SLICE_TYPE = {
    "basal": (
        (1, "basal anterior"),
        (2, "basal anteroseptal"),
        (3, "basal inferoseptal"),
        (4, "basal inferior"),
        (5, "basal inferolateral"),
        (6, "basal anterolateral"),
    ),
    "mid": (
        (7, "mid anterior"),
        (8, "mid anteroseptal"),
        (9, "mid inferoseptal"),
        (10, "mid inferior"),
        (11, "mid inferolateral"),
        (12, "mid anterolateral"),
    ),
    "apical": (
        (13, "apical anterior"),
        (14, "apical septal"),
        (15, "apical inferior"),
        (16, "apical lateral"),
    ),
}

OUTPUT_COLUMNS = [
    "patient_id",
    "slice_index",
    "slice_type",
    "aha_sector_number",
    "aha_sector_name",
    "diagnosis",
    "notes",
]


@dataclass(frozen=True, slots=True)
class WorkbookDiagnosisRow:
    patient_id: str
    slice_index: int
    slice_type: str
    video_name: str
    labels: tuple[str, ...]
    excel_row: int


def as_path(path_text: str | Path) -> Path:
    """Resolve pasted Windows or POSIX paths on the local machine."""
    raw = str(path_text).strip().strip('"').strip("'").replace("\\", "/")
    if len(raw) >= 2 and raw[1] == ":" and os.name != "nt":
        raw = f"/mnt/{raw[0].lower()}{raw[2:]}"
    return Path(raw).expanduser().resolve()


def normalize_patient_id(value: str | int | float) -> str:
    text = str(value).strip()
    if text.lower().startswith("patient"):
        digits = re.sub(r"\D", "", text)
    else:
        digits = str(int(float(text)))
    return f"patient{int(digits):04d}"


def normalize_slice_type(value: str) -> str:
    text = value.strip().lower()
    aliases = {"base": "basal", "middle": "mid", "apex": "apical"}
    text = aliases.get(text, text)
    if text not in SEGMENTS_BY_SLICE_TYPE:
        raise ValueError(f"Unsupported slice type {value!r}; expected basal, mid, or apical.")
    return text


def normalize_label_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value.strip().lower())
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def split_diagnosis_and_notes(label: str) -> tuple[str, str]:
    """Map workbook labels to the established diagnosis.csv diagnosis/notes columns."""
    text = normalize_label_text(label)
    if text == "normal":
        return "Normal", ""
    if text == "disincronia":
        return "Dyssynchrony", ""
    if text == "acinesia":
        return "Akinesia", ""
    if text == "discinesia":
        return "Dyskinesia", ""
    if text.startswith("hipocinesia"):
        severity = text.removeprefix("hipocinesia").strip()
        notes_by_severity = {
            "leve": "Light",
            "moderada": "Moderate",
            "severa": "Severe",
        }
        if severity not in notes_by_severity:
            raise ValueError(f"Unsupported hypokinesia severity in label {label!r}.")
        return "Hypokinesia", notes_by_severity[severity]
    raise ValueError(f"Unsupported ground-truth label {label!r}.")


def cell_column(cell_ref: str) -> str:
    return "".join(ch for ch in cell_ref if ch.isalpha())


def read_xlsx_first_sheet(path: Path) -> list[dict[str, str]]:
    """Read the first XLSX sheet as sparse Excel-column dictionaries without third-party deps."""
    with ZipFile(path) as archive:
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for si in root.findall(f"{XLSX_NS}si"):
                shared_strings.append(
                    "".join((node.text or "") for node in si.findall(f".//{XLSX_NS}t"))
                )

        sheet = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
        rows: list[dict[str, str]] = []
        for row in sheet.findall(f".//{XLSX_NS}row"):
            values: dict[str, str] = {}
            for cell in row.findall(f"{XLSX_NS}c"):
                column = cell_column(cell.attrib.get("r", ""))
                cell_type = cell.attrib.get("t")
                value_node = cell.find(f"{XLSX_NS}v")
                inline_node = cell.find(f"{XLSX_NS}is/{XLSX_NS}t")

                if inline_node is not None:
                    value = inline_node.text or ""
                elif value_node is None or value_node.text is None:
                    value = ""
                else:
                    value = value_node.text
                    if cell_type == "s":
                        value = shared_strings[int(value)]
                values[column] = value
            if values:
                rows.append(values)
        return rows


def parse_video_name(video_name: str) -> tuple[str, int]:
    match = re.search(r"(patient\d+)_slice(\d+)", video_name)
    if not match:
        raise ValueError(f"Could not parse patient id and slice index from video_name {video_name!r}.")
    return normalize_patient_id(match.group(1)), int(match.group(2))


def parse_workbook_rows(workbook_path: Path) -> list[WorkbookDiagnosisRow]:
    raw_rows = read_xlsx_first_sheet(workbook_path)
    if not raw_rows:
        raise ValueError(f"Workbook has no readable rows: {workbook_path}")

    parsed: list[WorkbookDiagnosisRow] = []
    for excel_index, row in enumerate(raw_rows[1:], start=2):
        if not row.get("A"):
            continue
        patient_id = normalize_patient_id(row["A"])
        slice_type = normalize_slice_type(row.get("B", ""))
        video_name = row.get("C", "").strip()
        video_patient_id, slice_index = parse_video_name(video_name)
        if video_patient_id != patient_id:
            raise ValueError(
                f"Workbook row {excel_index}: ID {patient_id} does not match video {video_name}."
            )

        labels = ast.literal_eval(row.get("D", ""))
        if not isinstance(labels, list) or not all(isinstance(item, str) for item in labels):
            raise ValueError(f"Workbook row {excel_index}: ground_truth must be a list of strings.")
        expected_count = len(SEGMENTS_BY_SLICE_TYPE[slice_type])
        if len(labels) != expected_count:
            raise ValueError(
                f"Workbook row {excel_index}: {slice_type} expects {expected_count} labels, "
                f"got {len(labels)}."
            )
        for label in labels:
            split_diagnosis_and_notes(label)

        parsed.append(
            WorkbookDiagnosisRow(
                patient_id=patient_id,
                slice_index=slice_index,
                slice_type=slice_type,
                video_name=video_name,
                labels=tuple(labels),
                excel_row=excel_index,
            )
        )
    return parsed


def discover_patient_ids(dataset_dir: Path) -> set[str]:
    return {
        normalize_patient_id(path.name)
        for path in dataset_dir.glob("patient*.nii.gz")
        if path.is_file()
    }


def validate_patient_coverage(patient_ids: set[str], workbook_rows: list[WorkbookDiagnosisRow]) -> None:
    workbook_patient_ids = {row.patient_id for row in workbook_rows}
    missing = sorted(patient_ids - workbook_patient_ids)
    extra = sorted(workbook_patient_ids - patient_ids)
    if missing or extra:
        details = []
        if missing:
            details.append("missing workbook entries for: " + ", ".join(missing))
        if extra:
            details.append("workbook entries without NIfTI files: " + ", ".join(extra))
        raise ValueError("Patient coverage mismatch; " + "; ".join(details))

    seen: set[tuple[str, int, str]] = set()
    duplicates: list[tuple[str, int, str]] = []
    for row in workbook_rows:
        key = (row.patient_id, row.slice_index, row.slice_type)
        if key in seen:
            duplicates.append(key)
        seen.add(key)
    if duplicates:
        duplicate_text = ", ".join(f"{pid}/slice{idx:02d}/{stype}" for pid, idx, stype in duplicates)
        raise ValueError(f"Duplicate workbook rows for the same patient/slice/type: {duplicate_text}")


def build_diagnosis_rows(workbook_rows: list[WorkbookDiagnosisRow]) -> list[dict[str, str | int]]:
    output_rows: list[dict[str, str | int]] = []
    for row in workbook_rows:
        for (segment_number, segment_name), label in zip(
            SEGMENTS_BY_SLICE_TYPE[row.slice_type],
            row.labels,
        ):
            diagnosis, notes = split_diagnosis_and_notes(label)
            output_rows.append(
                {
                    "patient_id": row.patient_id,
                    "slice_index": row.slice_index,
                    "slice_type": row.slice_type.title() if row.slice_type != "mid" else "Mid",
                    "aha_sector_number": segment_number,
                    "aha_sector_name": segment_name,
                    "diagnosis": diagnosis,
                    "notes": notes,
                }
            )
    return output_rows


def write_diagnosis_csv(path: Path, rows: list[dict[str, str | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    config = load_cardionet_config()
    script_cfg = config.scripts.build_cardiogpt_diagnosis_csv

    parser = argparse.ArgumentParser(
        description="Validate CardioGPT data_oishik patients against the XLSX labels and rebuild diagnosis.csv."
    )
    parser.add_argument("--content-dir", default=str(script_cfg.content_dir))
    parser.add_argument("--dataset-dirname", default=str(script_cfg.dataset_dirname))
    parser.add_argument("--workbook-name", default=str(script_cfg.workbook_name))
    parser.add_argument("--output-name", default=str(script_cfg.output_name))
    parser.add_argument("--dry-run", action="store_true", help="Validate and print counts without writing CSV.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    content_dir = as_path(args.content_dir)
    dataset_dir = content_dir / args.dataset_dirname
    workbook_path = content_dir / args.workbook_name
    output_path = dataset_dir / args.output_name

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    if not workbook_path.exists():
        raise FileNotFoundError(f"Workbook not found: {workbook_path}")

    patient_ids = discover_patient_ids(dataset_dir)
    workbook_rows = parse_workbook_rows(workbook_path)
    validate_patient_coverage(patient_ids, workbook_rows)
    diagnosis_rows = build_diagnosis_rows(workbook_rows)

    print("Dataset directory:", dataset_dir)
    print("Workbook:", workbook_path)
    print("NIfTI patients:", len(patient_ids))
    print("Workbook labelled slice rows:", len(workbook_rows))
    print("Diagnosis CSV rows to write:", len(diagnosis_rows))
    print("Output:", output_path)

    if args.dry_run:
        print("Dry run only; no file written.")
        return

    write_diagnosis_csv(output_path, diagnosis_rows)
    print("Wrote diagnosis CSV:", output_path)


if __name__ == "__main__":
    main()
