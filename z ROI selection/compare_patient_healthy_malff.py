import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t


DEFAULT_HEALTHY_NPZ = Path(r"D:\predata\npz_data\healthy_mALFF_AAL3_mean.npz")
DEFAULT_PATIENT_NPZ = Path(r"D:\predata\npz_data\patient_mALFF_AAL3_mean.npz")
DEFAULT_OUTPUT_CSV = Path(r"D:\predata\patient_vs_healthy_large_diff_rois.csv")
DEFAULT_VALUES_OUTPUT_CSV = Path(
    r"D:\predata\patient_vs_healthy_selected_rois_malff_grubbs.csv"
)


def grubbs_critical_value(n: int, alpha: float = 0.05) -> float:
    if n < 3:
        return np.inf

    t_value = t.ppf(1 - alpha / (2 * n), n - 2)
    return ((n - 1) / np.sqrt(n)) * np.sqrt(t_value**2 / (n - 2 + t_value**2))


def grubbs_replace_row(row: np.ndarray, alpha: float = 0.05) -> tuple[np.ndarray, int]:
    cleaned = np.asarray(row, dtype=float).copy()
    replaced_count = 0

    finite_count = np.isfinite(cleaned).sum()
    max_iter = max(0, finite_count - 2)

    for _ in range(max_iter):
        finite_idx = np.where(np.isfinite(cleaned))[0]
        values = cleaned[finite_idx]
        n = len(values)

        if n < 3:
            break

        mean_value = np.mean(values)
        std_value = np.std(values, ddof=1)
        if std_value == 0 or not np.isfinite(std_value):
            break

        deviations = np.abs(values - mean_value)
        outlier_local_idx = int(np.argmax(deviations))
        g_stat = deviations[outlier_local_idx] / std_value
        g_critical = grubbs_critical_value(n, alpha)

        if g_stat <= g_critical:
            break

        outlier_global_idx = finite_idx[outlier_local_idx]
        other_values = np.delete(values, outlier_local_idx)
        cleaned[outlier_global_idx] = np.mean(other_values)
        replaced_count += 1

    return cleaned, replaced_count


def grubbs_replace_matrix(matrix: np.ndarray, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(matrix, dtype=float)
    cleaned = np.empty_like(matrix, dtype=float)
    replaced_counts = np.zeros(matrix.shape[0], dtype=int)

    for roi_idx in range(matrix.shape[0]):
        cleaned[roi_idx], replaced_counts[roi_idx] = grubbs_replace_row(
            matrix[roi_idx], alpha=alpha
        )

    return cleaned, replaced_counts


def load_npz_matrix(npz_path: Path) -> tuple[np.ndarray, np.ndarray]:
    npz_data = np.load(npz_path, allow_pickle=True)
    if "data" in npz_data:
        matrix = npz_data["data"]
    elif "meanvalue_matrix" in npz_data:
        matrix = npz_data["meanvalue_matrix"]
    else:
        raise KeyError(f"{npz_path} does not contain data or meanvalue_matrix")

    if "roi_ids" not in npz_data:
        raise KeyError(f"{npz_path} does not contain roi_ids")

    return np.asarray(matrix, dtype=float), np.asarray(npz_data["roi_ids"], dtype=int)


def load_subject_ids(npz_path: Path, matrix: np.ndarray) -> np.ndarray:
    """Load and validate the sample IDs corresponding to matrix columns."""
    npz_data = np.load(npz_path, allow_pickle=True)
    if "subject_ids" not in npz_data:
        raise KeyError(f"{npz_path} does not contain subject_ids")

    subject_ids = np.asarray(npz_data["subject_ids"]).astype(str)
    if subject_ids.ndim != 1 or len(subject_ids) != matrix.shape[1]:
        raise ValueError(
            f"{npz_path} subject_ids count does not match matrix columns: "
            f"subject_ids={subject_ids.shape}, columns={matrix.shape[1]}"
        )
    if pd.Series(subject_ids).duplicated().any():
        raise ValueError(f"{npz_path} contains duplicate subject_ids")

    return subject_ids


def build_selected_values_dataframe(
    healthy_cleaned: np.ndarray,
    patient_cleaned: np.ndarray,
    healthy_subject_ids: np.ndarray,
    patient_subject_ids: np.ndarray,
    selected_roi_indices: np.ndarray,
    selected_roi_ids: np.ndarray,
    healthy_group_name: str = "healthy",
    patient_group_name: str = "patient",
) -> pd.DataFrame:
    """Combine corrected values for selected ROIs into one sample-level table."""
    roi_columns = [f"roi_{int(roi_id)}" for roi_id in selected_roi_ids]

    healthy_values = pd.DataFrame(
        healthy_cleaned[selected_roi_indices].T,
        columns=roi_columns,
    )
    healthy_values.insert(0, "group", healthy_group_name)
    healthy_values.insert(0, "subject_id", healthy_subject_ids)

    patient_values = pd.DataFrame(
        patient_cleaned[selected_roi_indices].T,
        columns=roi_columns,
    )
    patient_values.insert(0, "group", patient_group_name)
    patient_values.insert(0, "subject_id", patient_subject_ids)

    return pd.concat([healthy_values, patient_values], ignore_index=True)


def compare_groups(
    healthy_npz: Path,
    patient_npz: Path,
    output_csv: Path,
    alpha: float = 0.05,
    std_times: float = 2.0,
    values_output_csv: Path | None = None,
) -> pd.DataFrame:
    healthy_matrix, healthy_roi_ids = load_npz_matrix(healthy_npz)
    patient_matrix, patient_roi_ids = load_npz_matrix(patient_npz)
    healthy_subject_ids = load_subject_ids(healthy_npz, healthy_matrix)
    patient_subject_ids = load_subject_ids(patient_npz, patient_matrix)

    if healthy_matrix.shape[0] != patient_matrix.shape[0]:
        raise ValueError(
            f"ROI count mismatch: healthy={healthy_matrix.shape[0]}, "
            f"patient={patient_matrix.shape[0]}"
        )
    if not np.array_equal(healthy_roi_ids, patient_roi_ids):
        raise ValueError("The two NPZ files have different roi_ids")

    healthy_cleaned, healthy_outlier_counts = grubbs_replace_matrix(
        healthy_matrix, alpha=alpha
    )
    patient_cleaned, patient_outlier_counts = grubbs_replace_matrix(
        patient_matrix, alpha=alpha
    )

    healthy_mean = np.nanmean(healthy_cleaned, axis=1)
    patient_mean = np.nanmean(patient_cleaned, axis=1)
    difference = patient_mean - healthy_mean

    diff_mean = np.mean(difference)
    diff_std = np.std(difference)
    if diff_std == 0:
        raise ValueError("All ROI differences are identical, so z values cannot be computed")

    z_value = (difference - diff_mean) / diff_std
    selected_mask = np.abs(z_value) > std_times

    result = pd.DataFrame(
        {
            "roi_id": healthy_roi_ids[selected_mask],
            "healthy_mean": healthy_mean[selected_mask],
            "patient_mean": patient_mean[selected_mask],
            "difference_patient_minus_healthy": difference[selected_mask],
            "z_value": z_value[selected_mask],
            "healthy_grubbs_replaced_count": healthy_outlier_counts[selected_mask],
            "patient_grubbs_replaced_count": patient_outlier_counts[selected_mask],
        }
    )
    result = result.sort_values("z_value", key=lambda col: np.abs(col), ascending=False)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=False, encoding="utf-8-sig")

    if values_output_csv is None:
        values_output_csv = output_csv.with_name(
            f"{output_csv.stem}_malff_values.csv"
        )

    # Map the sorted summary ROI IDs back to their rows in the original matrices.
    roi_index_by_id = {int(roi_id): index for index, roi_id in enumerate(healthy_roi_ids)}
    selected_roi_ids = result["roi_id"].to_numpy(dtype=int)
    selected_roi_indices = np.array(
        [roi_index_by_id[int(roi_id)] for roi_id in selected_roi_ids],
        dtype=int,
    )
    values_result = build_selected_values_dataframe(
        healthy_cleaned=healthy_cleaned,
        patient_cleaned=patient_cleaned,
        healthy_subject_ids=healthy_subject_ids,
        patient_subject_ids=patient_subject_ids,
        selected_roi_indices=selected_roi_indices,
        selected_roi_ids=selected_roi_ids,
    )
    values_output_csv.parent.mkdir(parents=True, exist_ok=True)
    values_result.to_csv(values_output_csv, index=False, encoding="utf-8-sig")

    print(f"Healthy matrix: {healthy_matrix.shape}")
    print(f"Patient matrix: {patient_matrix.shape}")
    print(f"Difference mean: {diff_mean:.8f}")
    print(f"Difference std: {diff_std:.8f}")
    print(f"Threshold: abs(z_value) > {std_times:g}")
    print(f"Selected ROI count: {len(result)}")
    print(f"Saved: {output_csv}")
    print(f"Saved corrected values for {len(values_result)} samples: {values_output_csv}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare patient and healthy mALFF ROI means after Grubbs outlier replacement."
    )
    parser.add_argument("--healthy", type=Path, default=DEFAULT_HEALTHY_NPZ)
    parser.add_argument("--patient", type=Path, default=DEFAULT_PATIENT_NPZ)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument(
        "--values-output",
        type=Path,
        default=DEFAULT_VALUES_OUTPUT_CSV,
        help="Output CSV for Grubbs-corrected values of selected ROIs for every sample.",
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--std-times", type=float, default=2.0)
    args = parser.parse_args()

    compare_groups(
        healthy_npz=args.healthy,
        patient_npz=args.patient,
        output_csv=args.output,
        values_output_csv=args.values_output,
        alpha=args.alpha,
        std_times=args.std_times,
    )


if __name__ == "__main__":
    main()
