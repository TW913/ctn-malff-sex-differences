from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PARTICIPANT_FILE = "参与者信息总表.xlsx"
PAIN_COLUMN = "Pain_severity (average score)"
DISEASE_DURATION_COLUMN = "Disease_duration (years)"
BMI_COLUMN = "BMI"
MALFF_LONG_FILE = "all_mALFF_nuisanceRegressed_AAL3_mean_long.csv"
MALFF_WIDE_FILE = "patient_male_vs_female_all_samples_malff_grubbs.csv"


def extract_subject_id(path: Path) -> str:
    match = re.search(r"(sub-\d+)", path.name)
    if not match:
        raise ValueError(f"Could not extract subject id from {path}")
    return match.group(1)


def select_participant_files(
    files: Iterable[Path], participant_ids: Iterable[str]
) -> tuple[dict[str, Path], list[str], list[str]]:
    participant_id_set = set(participant_ids)
    by_subject = {extract_subject_id(path): path for path in files}
    selected = {
        subject_id: by_subject[subject_id]
        for subject_id in sorted(participant_id_set)
        if subject_id in by_subject
    }
    extra = sorted(set(by_subject) - participant_id_set)
    missing = sorted(participant_id_set - set(by_subject))
    return selected, extra, missing


def flatten_upper_triangle(matrix: pd.DataFrame) -> pd.Series:
    values = matrix.to_numpy(dtype=float)
    row_index, col_index = np.triu_indices_from(values, k=1)
    edge_names = [
        f"{matrix.index[row]}__{matrix.columns[col]}"
        for row, col in zip(row_index, col_index)
    ]
    return pd.Series(values[row_index, col_index], index=edge_names)


def load_participants(data_dir: Path) -> pd.DataFrame:
    participants = pd.read_excel(data_dir / PARTICIPANT_FILE)
    required_columns = [
        "BIDS_ID",
        "sex",
        "age",
        PAIN_COLUMN,
        DISEASE_DURATION_COLUMN,
        BMI_COLUMN,
        "Pain_side",
    ]
    missing = [column for column in required_columns if column not in participants.columns]
    if missing:
        raise ValueError(f"Missing required participant columns: {missing}")

    participants = participants.copy()
    participants["BIDS_ID"] = participants["BIDS_ID"].astype(str)
    participants["pain_severity"] = pd.to_numeric(participants[PAIN_COLUMN], errors="coerce")
    participants["disease_duration_years"] = pd.to_numeric(
        participants[DISEASE_DURATION_COLUMN], errors="coerce"
    )
    participants["BMI_numeric"] = pd.to_numeric(participants[BMI_COLUMN], errors="coerce")
    participants["pain_group"] = pd.cut(
        participants["pain_severity"],
        bins=[0, 3, 6, 10],
        labels=["mild_1_3", "moderate_4_6", "severe_7_10"],
        include_lowest=True,
    ).astype(str)
    return participants


def build_design_matrix(participants: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "BIDS_ID",
        "sex",
        "age",
        "pain_severity",
        "pain_group",
        "disease_duration_years",
        "BMI_numeric",
        "Pain_side",
        "Pain_type",
        "Sindou_grade",
    ]
    available_columns = [column for column in columns if column in participants.columns]
    return participants[available_columns].copy()


def align_malff_participants(
    table: pd.DataFrame, subject_column: str, participant_ids: list[str]
) -> tuple[pd.DataFrame, list[str], list[str]]:
    table = table.copy()
    if table[subject_column].isna().any():
        raise ValueError(f"mALFF table contains empty {subject_column} values")

    table[subject_column] = table[subject_column].astype(str)
    duplicate_ids = sorted(
        table.loc[table[subject_column].duplicated(), subject_column].unique()
    )
    if duplicate_ids:
        raise ValueError(f"mALFF table contains duplicate subject ids: {duplicate_ids}")

    source_ids = set(table[subject_column])
    participant_id_set = set(participant_ids)
    extra = sorted(source_ids - participant_id_set)
    missing = sorted(participant_id_set - source_ids)

    table = table.set_index(subject_column).reindex(participant_ids)
    table.index.name = "BIDS_ID"
    return table.reset_index(), extra, missing


def build_malff_wide_with_qc(
    data_dir: Path, participant_ids: list[str]
) -> tuple[pd.DataFrame, list[str], list[str]]:
    stats_dir = data_dir / "nii_stats"
    long_path = stats_dir / MALFF_LONG_FILE
    wide_path = stats_dir / MALFF_WIDE_FILE

    if long_path.exists():
        long = pd.read_csv(long_path)
        required_columns = {"Subject", "ROI", "MeanValue"}
        missing_columns = sorted(required_columns - set(long.columns))
        if missing_columns:
            raise ValueError(
                f"mALFF long table is missing required columns: {missing_columns}"
            )
        long["roi"] = "roi_" + long["ROI"].astype(str)
        wide = long.pivot(index="Subject", columns="roi", values="MeanValue")
        wide.columns.name = None
        wide = wide.reset_index()
        return align_malff_participants(wide, "Subject", participant_ids)

    if wide_path.exists():
        wide = pd.read_csv(wide_path)
        if "subject_id" not in wide.columns:
            raise ValueError(f"mALFF wide table is missing subject_id: {wide_path}")
        roi_columns = sorted(column for column in wide.columns if column.startswith("roi_"))
        if not roi_columns:
            raise ValueError(f"mALFF wide table has no ROI columns: {wide_path}")
        return align_malff_participants(
            wide[["subject_id", *roi_columns]], "subject_id", participant_ids
        )

    raise FileNotFoundError(
        "Could not find an mALFF input table. Expected either "
        f"{long_path.name} or {wide_path.name} in {stats_dir}"
    )


def build_malff_wide(data_dir: Path, participant_ids: list[str]) -> pd.DataFrame:
    malff, _, _ = build_malff_wide_with_qc(data_dir, participant_ids)
    return malff


def validate_matrix(matrix: pd.DataFrame, path: Path) -> None:
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Matrix is not square: {path}")
    if list(matrix.index) != list(matrix.columns):
        raise ValueError(f"Matrix row/column labels differ: {path}")
    values = matrix.to_numpy(dtype=float)
    if not np.allclose(values, values.T, atol=1e-8, equal_nan=True):
        raise ValueError(f"Matrix is not symmetric: {path}")
    if np.isnan(values).any():
        raise ValueError(f"Matrix contains NaN values: {path}")


def graph_metrics_from_matrix(matrix: pd.DataFrame) -> pd.Series:
    values = matrix.to_numpy(dtype=float).copy()
    np.fill_diagonal(values, np.nan)
    valid_edge_count = int(np.count_nonzero(~np.isnan(values)))

    metrics = {
        "mean_fisher_z": float(np.nanmean(values)),
        "mean_abs_fisher_z": float(np.nanmean(np.abs(values))),
        "positive_edge_fraction": float(np.count_nonzero(values > 0) / valid_edge_count),
        "negative_edge_fraction": float(np.count_nonzero(values < 0) / valid_edge_count),
    }
    node_strength = np.nanmean(values, axis=1)
    abs_node_strength = np.nanmean(np.abs(values), axis=1)
    for roi, value in zip(matrix.index, node_strength):
        metrics[f"node_strength_{roi}"] = float(value)
    for roi, value in zip(matrix.index, abs_node_strength):
        metrics[f"abs_node_strength_{roi}"] = float(value)
    return pd.Series(metrics)


def build_connectivity_tables(
    matrix_files: dict[str, Path], participant_ids: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    edge_rows = []
    metric_rows = []
    reference_labels = None

    for subject_id in participant_ids:
        matrix = pd.read_csv(matrix_files[subject_id], index_col=0)
        validate_matrix(matrix, matrix_files[subject_id])
        labels = list(matrix.index)
        if reference_labels is None:
            reference_labels = labels
        elif labels != reference_labels:
            raise ValueError(f"ROI labels differ for {matrix_files[subject_id]}")

        edge_row = flatten_upper_triangle(matrix)
        edge_row["BIDS_ID"] = subject_id
        edge_rows.append(edge_row)

        metric_row = graph_metrics_from_matrix(matrix)
        metric_row["BIDS_ID"] = subject_id
        metric_rows.append(metric_row)

    edges = pd.DataFrame(edge_rows).set_index("BIDS_ID").reindex(participant_ids).reset_index()
    metrics = pd.DataFrame(metric_rows).set_index("BIDS_ID").reindex(participant_ids).reset_index()
    qc = {
        "roi_count": len(reference_labels or []),
        "edge_count": int(edges.shape[1] - 1),
        "roi_labels": reference_labels or [],
    }
    return edges, metrics, qc


def write_qc_report(
    output_dir: Path,
    participants: pd.DataFrame,
    design: pd.DataFrame,
    malff: pd.DataFrame,
    edges: pd.DataFrame,
    metrics: pd.DataFrame,
    file_qc: dict[str, object],
) -> None:
    pain_counts = design["pain_group"].value_counts().sort_index().to_dict()
    sex_counts = design["sex"].value_counts(dropna=False).sort_index().to_dict()
    sex_by_pain = pd.crosstab(design["sex"], design["pain_group"]).to_dict()
    summary = {
        "participant_count": int(len(participants)),
        "analysis_participant_count": int(len(design)),
        "sex_counts": {str(key): int(value) for key, value in sex_counts.items()},
        "pain_group_counts": {str(key): int(value) for key, value in pain_counts.items()},
        "sex_by_pain": {
            str(group): {str(sex): int(count) for sex, count in values.items()}
            for group, values in sex_by_pain.items()
        },
        "malff_shape": list(malff.shape),
        "connectivity_edges_shape": list(edges.shape),
        "graph_metrics_shape": list(metrics.shape),
        **file_qc,
    }
    (output_dir / "qc_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    pain_stats = design["pain_severity"].describe()
    report = [
        "# MRI Network Model Input QC",
        "",
        "## Cohort",
        f"- Participants in clinical table: {len(participants)}",
        f"- Participants retained for analysis: {len(design)}",
        f"- Sex counts: {summary['sex_counts']}",
        f"- Pain severity: mean {pain_stats['mean']:.2f}, "
        f"median {pain_stats['50%']:.2f}, range {pain_stats['min']:.0f}-{pain_stats['max']:.0f}",
        f"- Pain groups: {summary['pain_group_counts']}",
        "",
        "## Imaging Inputs",
        f"- mALFF wide table shape: {malff.shape[0]} rows x {malff.shape[1]} columns",
        f"- Fisher-z edge table shape: {edges.shape[0]} rows x {edges.shape[1]} columns",
        f"- Graph metrics table shape: {metrics.shape[0]} rows x {metrics.shape[1]} columns",
        f"- ROI count in matrices: {summary['roi_count']}",
        f"- Unique upper-triangle edges: {summary['edge_count']}",
        "",
        "## File Matching",
        f"- Missing participant mALFF files: {len(file_qc['missing_malff_subjects'])}",
        f"- Missing participant Fisher-z matrices: {len(file_qc['missing_matrix_subjects'])}",
        f"- Extra mALFF subject labels not in clinical table: {len(file_qc['extra_malff_subjects'])}",
        f"- Extra Fisher-z subject labels not in clinical table: {len(file_qc['extra_matrix_subjects'])}",
        "",
        "## Modeling Note",
        "- Prefer Fisher-z matrices for inferential connectivity statistics.",
        "- Use one joint model for sex, pain severity, and sex:pain because sex and pain are imbalanced.",
        "- Treat sex coding as unknown until 0/1 labels are confirmed from study metadata.",
    ]
    (output_dir / "qc_summary.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def write_model_notes(output_dir: Path) -> None:
    notes = """# Suggested Starting Models

Use these files as model inputs:

- `participants_for_analysis.csv`: clinical covariates and pain groups.
- `mALFF_wide.csv`: one row per participant, one column per ROI mALFF value.
- `connectivity_edges_fisher_z_wide.csv.gz`: one row per participant, one column per unique Fisher-z edge.
- `graph_metrics_wide.csv`: global and node-level connectivity summaries.

Primary covariate model:

```text
brain_metric ~ sex + pain_severity + sex:pain_severity + age + disease_duration_years + BMI_numeric + Pain_side
```

Recommended sequence:

1. Start with graph metrics and network/ROI summaries for interpretable screening.
2. Run ROI-wise mALFF GLMs with FDR correction.
3. Run edge-wise GLMs or NBS on Fisher-z matrices for connectivity-level inference.
4. Use pain severity as continuous in the main analysis; keep pain groups for descriptive plots or sensitivity checks.
"""
    (output_dir / "model_formula_notes.md").write_text(notes, encoding="utf-8")


def build_outputs(data_dir: Path, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    participants = load_participants(data_dir)
    participant_ids = participants["BIDS_ID"].tolist()

    matrix_files, extra_matrix, missing_matrix = select_participant_files(
        (data_dir / "matrices").glob("sub-*_fisher_z.csv"),
        participant_ids,
    )
    malff, extra_malff, missing_malff = build_malff_wide_with_qc(
        data_dir, participant_ids
    )
    if missing_matrix or missing_malff:
        raise ValueError(
            "Missing imaging files for clinical participants: "
            f"matrix={missing_matrix}, mALFF={missing_malff}"
        )

    design = build_design_matrix(participants)
    edges, metrics, matrix_qc = build_connectivity_tables(matrix_files, participant_ids)

    design.to_csv(output_dir / "participants_for_analysis.csv", index=False)
    malff.to_csv(output_dir / "mALFF_wide.csv", index=False)
    edges.to_csv(output_dir / "connectivity_edges_fisher_z_wide.csv.gz", index=False)
    metrics.to_csv(output_dir / "graph_metrics_wide.csv", index=False)

    file_qc = {
        "extra_matrix_subjects": extra_matrix,
        "missing_matrix_subjects": missing_matrix,
        "extra_malff_subjects": extra_malff,
        "missing_malff_subjects": missing_malff,
        **matrix_qc,
    }
    write_qc_report(output_dir, participants, design, malff, edges, metrics, file_qc)
    write_model_notes(output_dir)

    return {
        "participants": output_dir / "participants_for_analysis.csv",
        "malff": output_dir / "mALFF_wide.csv",
        "edges": output_dir / "connectivity_edges_fisher_z_wide.csv.gz",
        "graph_metrics": output_dir / "graph_metrics_wide.csv",
        "qc_markdown": output_dir / "qc_summary.md",
        "qc_json": output_dir / "qc_summary.json",
        "model_notes": output_dir / "model_formula_notes.md",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build MRI model input tables.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("analysis") / "model_inputs"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = build_outputs(args.data_dir, args.output_dir)
    print("Wrote model input files:")
    for name, path in outputs.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
