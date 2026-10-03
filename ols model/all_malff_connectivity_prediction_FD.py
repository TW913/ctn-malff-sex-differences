from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t as student_t_distribution


EFFECT_TERMS = ["sex", "pain_severity", "sex_x_pain"]
MODEL_COVARIATES = [
    "sex",
    "pain_severity",
    "age",
    "disease_duration_years",
    "BMI_numeric",
    "MeanFD",
]
DEFAULT_PLOT_TOP_N = 9


def normal_two_sided_p(t_value: float) -> float:
    return math.erfc(abs(float(t_value)) / math.sqrt(2.0))


def student_t_two_sided_p(t_value: float, df_resid: int) -> float:
    if np.isnan(t_value) or df_resid <= 0:
        return np.nan
    return float(2.0 * student_t_distribution.sf(abs(float(t_value)), df=df_resid))


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    q = np.full_like(p, np.nan, dtype=float)
    valid = np.isfinite(p)
    if not valid.any():
        return q
    valid_p = p[valid]
    order = np.argsort(valid_p)
    ranked = valid_p[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    restored = np.empty_like(valid_p)
    restored[order] = adjusted
    q[valid] = restored
    return q


def zscore(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    std = values.std(ddof=0)
    if pd.isna(std) or std == 0:
        return values * 0.0
    return (values - values.mean()) / std


def build_design(participants: pd.DataFrame) -> pd.DataFrame:
    design = pd.DataFrame(index=participants.index)
    design["intercept"] = 1.0
    design["sex"] = pd.to_numeric(participants["sex"], errors="coerce")
    design["pain_severity"] = zscore(participants["pain_severity"])
    design["sex_x_pain"] = design["sex"] * design["pain_severity"]
    design["age"] = zscore(participants["age"])
    design["disease_duration_years"] = zscore(participants["disease_duration_years"])
    design["BMI_numeric"] = zscore(participants["BMI_numeric"])
    design["MeanFD"] = zscore(participants["MeanFD"])
    return design.astype(float)


def fit_ols_for_outcome(y: np.ndarray, design: pd.DataFrame) -> dict[str, dict[str, float]]:
    design_values = design.to_numpy(dtype=float)
    valid = np.isfinite(y) & np.isfinite(design_values).all(axis=1)
    x = design_values[valid]
    y_valid = np.asarray(y[valid], dtype=float)
    n_obs = len(y_valid)
    rank = int(np.linalg.matrix_rank(x))
    if n_obs <= rank:
        raise ValueError("Not enough observations for the design matrix")
    beta, *_ = np.linalg.lstsq(x, y_valid, rcond=None)
    residuals = y_valid - x @ beta
    df_resid = n_obs - rank
    sigma2 = float((residuals @ residuals) / df_resid)
    covariance = sigma2 * np.linalg.pinv(x.T @ x)
    se = np.sqrt(np.diag(covariance))
    with np.errstate(divide="ignore", invalid="ignore"):
        t_values = beta / se
    results: dict[str, dict[str, float]] = {}
    critical = student_t_distribution.ppf(0.975, df=df_resid)
    for idx, term in enumerate(design.columns):
        t_value = float(t_values[idx])
        p_normal = normal_two_sided_p(t_value)
        results[term] = {
            "beta": float(beta[idx]),
            "se": float(se[idx]),
            "t": t_value,
            "p": p_normal,
            "p_normal": p_normal,
            "p_student_t": student_t_two_sided_p(t_value, df_resid),
            "n": int(n_obs),
            "df_resid": int(df_resid),
            "ci_lower": float(beta[idx] - critical * se[idx]),
            "ci_upper": float(beta[idx] + critical * se[idx]),
        }
    return results


def run_mass_univariate(
    outcomes: pd.DataFrame, participants: pd.DataFrame, design: pd.DataFrame
) -> pd.DataFrame:
    aligned = outcomes.set_index("BIDS_ID").loc[participants["BIDS_ID"]]
    rows = []
    for feature in aligned.columns:
        term_results = fit_ols_for_outcome(aligned[feature].to_numpy(dtype=float), design)
        for term in EFFECT_TERMS:
            row = {"feature": feature, "term": term}
            row.update(term_results[term])
            rows.append(row)
    results = pd.DataFrame(rows)
    results["q_fdr_normal"] = np.nan
    results["q_fdr_student_t"] = np.nan
    for term in EFFECT_TERMS:
        mask = results["term"] == term
        results.loc[mask, "q_fdr_normal"] = benjamini_hochberg(
            results.loc[mask, "p_normal"].to_numpy()
        )
        results.loc[mask, "q_fdr_student_t"] = benjamini_hochberg(
            results.loc[mask, "p_student_t"].to_numpy()
        )
    results["q_fdr"] = results["q_fdr_normal"]
    results["abs_t"] = results["t"].abs()
    return results.sort_values(["term", "q_fdr", "p", "feature"]).reset_index(drop=True)


def calculate_fit_metrics(
    outcomes: pd.DataFrame, participants: pd.DataFrame, design: pd.DataFrame
) -> pd.DataFrame:
    aligned = outcomes.set_index("BIDS_ID").loc[participants["BIDS_ID"]]
    design_values = design.to_numpy(dtype=float)
    rows = []
    for feature in aligned.columns:
        y = aligned[feature].to_numpy(dtype=float)
        valid = np.isfinite(y) & np.isfinite(design_values).all(axis=1)
        x_valid = design_values[valid]
        y_valid = y[valid]
        rank = int(np.linalg.matrix_rank(x_valid))
        if len(y_valid) <= rank:
            continue
        coefficients, *_ = np.linalg.lstsq(x_valid, y_valid, rcond=None)
        fitted = x_valid @ coefficients
        residuals = y_valid - fitted
        sse = float(np.sum(residuals**2))
        total = float(np.sum((y_valid - y_valid.mean()) ** 2))
        r2 = float(1.0 - sse / total) if total > 0 else np.nan
        df_resid = len(y_valid) - rank
        adjusted_r2 = (
            float(1.0 - (1.0 - r2) * (len(y_valid) - 1) / df_resid)
            if np.isfinite(r2)
            else np.nan
        )
        rows.append(
            {
                "roi": feature,
                "n": int(len(y_valid)),
                "model_rank": rank,
                "df_resid": int(df_resid),
                "r2": r2,
                "adjusted_r2": adjusted_r2,
                "rmse": float(np.sqrt(np.mean(residuals**2))),
                "mae": float(np.mean(np.abs(residuals))),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["adjusted_r2", "r2", "rmse", "roi"],
        ascending=[False, False, True, True],
        na_position="last",
    ).reset_index(drop=True)


def _fitted_values(
    outcomes: pd.DataFrame,
    participants: pd.DataFrame,
    design: pd.DataFrame,
    roi: str,
) -> tuple[np.ndarray, np.ndarray]:
    aligned = outcomes.set_index("BIDS_ID").loc[participants["BIDS_ID"]]
    observed = aligned[roi].to_numpy(dtype=float)
    design_values = design.to_numpy(dtype=float)
    valid = np.isfinite(observed) & np.isfinite(design_values).all(axis=1)
    x_valid = design_values[valid]
    observed = observed[valid]
    coefficients, *_ = np.linalg.lstsq(x_valid, observed, rcond=None)
    return observed, x_valid @ coefficients


def _save_panel_plot(
    outcomes: pd.DataFrame,
    participants: pd.DataFrame,
    design: pd.DataFrame,
    selected: pd.DataFrame,
    output_path: Path,
    title: str,
    annotated: bool,
    tiff: bool = False,
) -> None:
    if selected.empty:
        raise ValueError("No ROI is available for plotting")
    n_panels = len(selected)
    n_columns = min(3, n_panels)
    n_rows = int(math.ceil(n_panels / n_columns))
    figure, axes = plt.subplots(
        n_rows, n_columns, figsize=(4.6 * n_columns, 4.2 * n_rows), squeeze=False
    )
    flat_axes = axes.ravel()
    for axis, (_, metric) in zip(flat_axes, selected.iterrows()):
        roi = metric["roi"]
        observed, fitted = _fitted_values(outcomes, participants, design, roi)
        axis.scatter(
            observed,
            fitted,
            s=52,
            alpha=0.85,
            color="#1f77b4",
            edgecolors="white",
            linewidths=0.6,
        )
        lower = float(min(observed.min(), fitted.min()))
        upper = float(max(observed.max(), fitted.max()))
        padding = max((upper - lower) * 0.06, 0.02)
        axis.plot(
            [lower - padding, upper + padding],
            [lower - padding, upper + padding],
            color="#555555",
            linestyle="--",
            linewidth=1.1,
        )
        axis.set_xlim(lower - padding, upper + padding)
        axis.set_ylim(lower - padding, upper + padding)
        axis.set_aspect("equal", adjustable="box")
        if annotated:
            axis.set_xlabel("Observed mALFF", fontweight="bold")
            axis.set_ylabel("Fitted mALFF", fontweight="bold")
            axis.set_title(str(roi), fontweight="bold")
            axis.text(
                0.97,
                0.97,
                f"Adjusted R2 = {metric['adjusted_r2']:.3f}\nRMSE = {metric['rmse']:.3f}",
                transform=axis.transAxes,
                ha="right",
                va="top",
                fontweight="bold",
            )
            axis.grid(alpha=0.22)
        else:
            axis.set_xlabel("")
            axis.set_ylabel("")
            axis.set_title("")
            axis.grid(False)
    for axis in flat_axes[n_panels:]:
        axis.set_visible(False)
    if annotated:
        figure.suptitle(title, fontsize=14, fontweight="bold", y=1.01)
    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=1200 if tiff else 300,
        format="tiff" if tiff else None,
        bbox_inches="tight",
    )
    plt.close(figure)


def save_plots(
    outcomes: pd.DataFrame,
    participants: pd.DataFrame,
    design: pd.DataFrame,
    roi_results: pd.DataFrame,
    fit_metrics: pd.DataFrame,
    output_dir: Path,
    top_n: int,
) -> None:
    top = fit_metrics.loc[np.isfinite(fit_metrics["adjusted_r2"])].head(top_n)
    _save_panel_plot(
        outcomes,
        participants,
        design,
        top,
        output_dir / "full_sample_glm_observed_vs_fitted_top_rois.png",
        "Full-sample GLM: Observed vs. In-sample Fitted mALFF",
        annotated=True,
    )

    significant = roi_results.loc[
        pd.to_numeric(roi_results["q_fdr_normal"], errors="coerce") < 0.05,
        ["feature", "q_fdr_normal"],
    ].rename(columns={"feature": "roi"})
    significant = significant.groupby("roi", as_index=False)["q_fdr_normal"].min()
    selected = fit_metrics.merge(significant, on="roi", how="inner").sort_values(
        ["adjusted_r2", "rmse", "roi"], ascending=[False, True, True]
    )
    if selected.empty:
        selected = fit_metrics.loc[np.isfinite(fit_metrics["adjusted_r2"])].head(top_n).copy()
        title = "Full-sample GLM: Top ROIs (no FDR q < 0.05 ROI)"
    else:
        title = "Full-sample GLM: Normal-approximation FDR q < 0.05 ROIs"
    _save_panel_plot(
        outcomes,
        participants,
        design,
        selected,
        output_dir / "full_sample_glm_normal_fdr_q_lt_0_05_annotated.tif",
        title,
        annotated=True,
        tiff=True,
    )
    _save_panel_plot(
        outcomes,
        participants,
        design,
        selected,
        output_dir / "full_sample_glm_normal_fdr_q_lt_0_05_ticks_only.tif",
        title,
        annotated=False,
        tiff=True,
    )


def align_outcomes(malff: pd.DataFrame, participants: pd.DataFrame) -> pd.DataFrame:
    if "BIDS_ID" not in malff.columns:
        raise ValueError("mALFF_wide.csv must contain a BIDS_ID column")
    if malff["BIDS_ID"].duplicated().any():
        raise ValueError("mALFF_wide.csv contains duplicate BIDS_ID values")
    feature_columns = [column for column in malff.columns if column != "BIDS_ID"]
    if not feature_columns:
        raise ValueError("mALFF_wide.csv has no ROI feature columns")
    indexed = malff.copy()
    indexed["BIDS_ID"] = indexed["BIDS_ID"].astype(str)
    indexed = indexed.set_index("BIDS_ID")
    participant_ids = participants["BIDS_ID"].astype(str)
    missing = sorted(set(participant_ids) - set(indexed.index))
    if missing:
        raise ValueError(f"mALFF_wide.csv is missing participant IDs: {missing}")
    aligned = indexed.loc[participant_ids, feature_columns].apply(
        pd.to_numeric, errors="coerce"
    )
    aligned.insert(0, "BIDS_ID", participant_ids.to_numpy())
    return aligned.reset_index(drop=True)


def add_motion_covariate(input_dir: Path, participants: pd.DataFrame) -> pd.DataFrame:
    motion_path = input_dir / "motion_metrics.csv"
    if not motion_path.exists():
        raise FileNotFoundError(f"MeanFD covariate file not found: {motion_path}")
    motion = pd.read_csv(motion_path)
    required = {"SubjectID", "MeanFD"}
    missing = sorted(required - set(motion.columns))
    if missing:
        raise ValueError(f"motion_metrics.csv is missing required columns: {missing}")
    motion = motion[["SubjectID", "MeanFD"]].copy()
    motion["SubjectID"] = motion["SubjectID"].astype(str)
    if motion["SubjectID"].duplicated().any():
        raise ValueError("motion_metrics.csv contains duplicate SubjectID values")
    motion["MeanFD"] = pd.to_numeric(motion["MeanFD"], errors="coerce")
    if motion["MeanFD"].isna().any():
        raise ValueError("motion_metrics.csv contains missing or non-numeric MeanFD values")
    motion_indexed = motion.set_index("SubjectID")
    participant_ids = set(participants["BIDS_ID"])
    motion_ids = set(motion_indexed.index)
    if participant_ids != motion_ids:
        raise ValueError(
            "MeanFD IDs do not match analysis participants. "
            f"Missing: {sorted(participant_ids - motion_ids)}; "
            f"extra: {sorted(motion_ids - participant_ids)}"
        )
    result = participants.copy()
    result["MeanFD"] = result["BIDS_ID"].map(motion_indexed["MeanFD"])
    return result


def run_analysis(input_dir: Path, output_dir: Path, plot_top_n: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    participants = pd.read_csv(input_dir / "participants_for_analysis.csv")
    if "BIDS_ID" not in participants.columns:
        raise ValueError("participants_for_analysis.csv must contain a BIDS_ID column")
    participants = participants.copy()
    participants["BIDS_ID"] = participants["BIDS_ID"].astype(str)
    if participants["BIDS_ID"].duplicated().any():
        raise ValueError("participants_for_analysis.csv contains duplicate BIDS_ID values")
    participants = add_motion_covariate(input_dir, participants)
    missing = sorted(set(MODEL_COVARIATES) - set(participants.columns))
    if missing:
        raise ValueError(f"Missing required covariate columns: {missing}")
    malff = align_outcomes(pd.read_csv(input_dir / "mALFF_wide.csv"), participants)
    design = build_design(participants)
    roi_results = run_mass_univariate(malff, participants, design)
    fit_metrics = calculate_fit_metrics(malff, participants, design)
    roi_results.to_csv(output_dir / "roi_malff_glm_results.csv", index=False)
    save_plots(malff, participants, design, roi_results, fit_metrics, output_dir, plot_top_n)
    print(f"Wrote roi_malff_glm_results.csv and plots to {output_dir}")


def parse_args() -> argparse.Namespace:
    script_directory = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Run the full-sample mALFF ROI GLM and save its plots."
    )
    parser.add_argument("--input-dir", type=Path, default=script_directory / "model_inputs")
    parser.add_argument("--output-dir", type=Path, default=script_directory / "FD results")
    parser.add_argument("--plot-top-n", type=int, default=DEFAULT_PLOT_TOP_N)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.plot_top_n < 1:
        raise ValueError("plot_top_n must be at least 1")
    run_analysis(args.input_dir, args.output_dir, args.plot_top_n)


if __name__ == "__main__":
    main()
