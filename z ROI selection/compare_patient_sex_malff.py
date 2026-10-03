import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from compare_patient_healthy_malff import (
    build_selected_values_dataframe,
    grubbs_replace_matrix,
    load_npz_matrix,
    load_subject_ids,
)


DEFAULT_FEMALE_NPZ = Path(r"D:\predata\npz_data\patient_female_mALFF_AAL3_mean.npz")
DEFAULT_MALE_NPZ = Path(r"D:\predata\npz_data\patient_male_mALFF_AAL3_mean.npz")
DEFAULT_OUTPUT_CSV = Path(r"D:\predata\patient_male_vs_female_large_diff_rois.csv")
DEFAULT_VALUES_OUTPUT_CSV = Path(
    r"D:\predata\patient_male_vs_female_selected_rois_malff_grubbs.csv"
)
DEFAULT_ALL_VALUES_OUTPUT_CSV = Path(
    r"D:\predata\格布斯检验\patient_male_vs_female_all_samples_malff_grubbs.csv"
)


def compare_sex_groups(
    female_npz: Path,
    male_npz: Path,
    output_csv: Path,
    alpha: float = 0.05,
    std_times: float = 2.0,
    values_output_csv: Path | None = None,
    all_values_output_csv: Path | None = DEFAULT_ALL_VALUES_OUTPUT_CSV,
) -> pd.DataFrame:
    female_matrix, female_roi_ids = load_npz_matrix(female_npz)
    male_matrix, male_roi_ids = load_npz_matrix(male_npz)
    female_subject_ids = load_subject_ids(female_npz, female_matrix)
    male_subject_ids = load_subject_ids(male_npz, male_matrix)

    if female_matrix.shape[0] != male_matrix.shape[0]:
        raise ValueError(
            f"ROI count mismatch: female={female_matrix.shape[0]}, "
            f"male={male_matrix.shape[0]}"
        )
    if not np.array_equal(female_roi_ids, male_roi_ids):
        raise ValueError("The two NPZ files have different roi_ids")

    female_cleaned, female_outlier_counts = grubbs_replace_matrix(
        female_matrix, alpha=alpha
    )
    male_cleaned, male_outlier_counts = grubbs_replace_matrix(
        male_matrix, alpha=alpha
    )

    female_mean = np.nanmean(female_cleaned, axis=1)
    male_mean = np.nanmean(male_cleaned, axis=1)
    difference = male_mean - female_mean

    diff_mean = np.mean(difference)
    diff_std = np.std(difference)
    if diff_std == 0:
        raise ValueError("All ROI differences are identical, so z values cannot be computed")

    z_value = (difference - diff_mean) / diff_std
    selected_mask = np.abs(z_value) > std_times

    direction = np.where(
        difference[selected_mask] > 0,
        "male_higher",
        "female_higher",
    )

    result = pd.DataFrame(
        {
            "roi_id": female_roi_ids[selected_mask],
            "female_mean": female_mean[selected_mask],
            "male_mean": male_mean[selected_mask],
            "difference_male_minus_female": difference[selected_mask],
            "z_value": z_value[selected_mask],
            "direction": direction,
            "female_grubbs_replaced_count": female_outlier_counts[selected_mask],
            "male_grubbs_replaced_count": male_outlier_counts[selected_mask],
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
    roi_index_by_id = {int(roi_id): index for index, roi_id in enumerate(female_roi_ids)}
    selected_roi_ids = result["roi_id"].to_numpy(dtype=int)
    selected_roi_indices = np.array(
        [roi_index_by_id[int(roi_id)] for roi_id in selected_roi_ids],
        dtype=int,
    )
    values_result = build_selected_values_dataframe(
        healthy_cleaned=female_cleaned,
        patient_cleaned=male_cleaned,
        healthy_subject_ids=female_subject_ids,
        patient_subject_ids=male_subject_ids,
        selected_roi_indices=selected_roi_indices,
        selected_roi_ids=selected_roi_ids,
        healthy_group_name="female",
        patient_group_name="male",
    )
    values_output_csv.parent.mkdir(parents=True, exist_ok=True)
    values_result.to_csv(values_output_csv, index=False, encoding="utf-8-sig")

    if all_values_output_csv is None:
        all_values_output_csv = DEFAULT_ALL_VALUES_OUTPUT_CSV

    # Export the corrected activation values for every ROI and every sample.
    all_roi_indices = np.arange(len(female_roi_ids), dtype=int)
    all_values_result = build_selected_values_dataframe(
        healthy_cleaned=female_cleaned,
        patient_cleaned=male_cleaned,
        healthy_subject_ids=female_subject_ids,
        patient_subject_ids=male_subject_ids,
        selected_roi_indices=all_roi_indices,
        selected_roi_ids=female_roi_ids,
        healthy_group_name="female",
        patient_group_name="male",
    )
    all_values_output_csv.parent.mkdir(parents=True, exist_ok=True)
    all_values_result.to_csv(all_values_output_csv, index=False, encoding="utf-8-sig")

    print(f"Female patient matrix: {female_matrix.shape}")
    print(f"Male patient matrix: {male_matrix.shape}")
    print(f"Difference mean: {diff_mean:.8f}")
    print(f"Difference std: {diff_std:.8f}")
    print(f"Threshold: abs(z_value) > {std_times:g}")
    print(f"Selected ROI count: {len(result)}")
    print(f"Saved: {output_csv}")
    print(f"Saved corrected values for {len(values_result)} samples: {values_output_csv}")
    print(
        f"Saved corrected values for all {len(all_values_result)} samples and "
        f"{len(female_roi_ids)} ROIs: {all_values_output_csv}"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare male and female patient mALFF ROI means after Grubbs outlier replacement."
    )
    parser.add_argument("--female", type=Path, default=DEFAULT_FEMALE_NPZ)
    parser.add_argument("--male", type=Path, default=DEFAULT_MALE_NPZ)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument(
        "--values-output",
        type=Path,
        default=DEFAULT_VALUES_OUTPUT_CSV,
        help="Output CSV for Grubbs-corrected values of selected ROIs for every sample.",
    )
    parser.add_argument(
        "--all-values-output",
        type=Path,
        default=DEFAULT_ALL_VALUES_OUTPUT_CSV,
        help="Output CSV for Grubbs-corrected values of all ROIs for every sample.",
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--std-times", type=float, default=2.0)
    args = parser.parse_args()

    compare_sex_groups(
        female_npz=args.female,
        male_npz=args.male,
        output_csv=args.output,
        values_output_csv=args.values_output,
        all_values_output_csv=args.all_values_output,
        alpha=args.alpha,
        std_times=args.std_times,
    )


if __name__ == "__main__":
    main()
