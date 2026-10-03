from pathlib import Path
import re

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t


BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "性别效应的稳健性.xlsx"
OUTPUT_FILE = BASE_DIR / "Figure_S3_forest.tif"


def configure_font() -> str:
    """Load Times New Roman and apply publication-sized defaults."""
    font_path = Path(r"C:\Windows\Fonts\times.ttf")
    font_name = "Times New Roman"
    if font_path.exists():
        fm.fontManager.addfont(font_path)
        font_name = fm.FontProperties(fname=font_path).get_name()

    plt.rcParams.update(
        {
            "font.family": font_name,
            "font.size": 11,
            "axes.labelsize": 12,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 11,
            "axes.unicode_minus": False,
            "axes.linewidth": 0.9,
        }
    )
    return font_name


def clean_model_label(value: str) -> str:
    """Remove degrees-of-freedom suffixes from displayed model labels."""
    label = str(value).strip()
    label = re.sub(r"\s*\(df\s*=\s*\d+\)\s*$", "", label, flags=re.IGNORECASE)
    return label


def load_and_prepare() -> pd.DataFrame:
    data = pd.read_excel(DATA_FILE, sheet_name=0)
    required = ["ROI", "Model", "Beta", "SE", "df", "q_t"]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Keep only models that were actually run; guard against future additions
    # of an explicitly excluded high-motion model.
    exclusion_pattern = r"high\s*[-_ ]?motion|motion\s*exclusion|exclusion\s*model"
    data = data.loc[
        ~data["Model"].astype(str).str.contains(
            exclusion_pattern, case=False, regex=True, na=False
        )
    ].copy()
    if data.empty:
        raise ValueError("No eligible models remain after excluding high-motion models.")

    numeric_columns = ["Beta", "SE", "df", "q_t"]
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    if data[required].isna().any().any():
        bad_columns = data[required].columns[data[required].isna().any()].tolist()
        raise ValueError(f"Missing or non-numeric values in: {bad_columns}")
    if (data["SE"] <= 0).any() or (data["df"] <= 0).any():
        raise ValueError("SE and df must be positive for all rows.")

    critical_value = t.ppf(0.975, data["df"])
    data["CI_lower"] = data["Beta"] - critical_value * data["SE"]
    data["CI_upper"] = data["Beta"] + critical_value * data["SE"]
    data["Model_label"] = data["Model"].map(clean_model_label)
    data["Combination"] = data["ROI"].astype(str) + " | " + data["Model_label"]
    return data


def main() -> None:
    configure_font()
    data = load_and_prepare()

    # Preserve spreadsheet row order so the displayed model sequence is auditable.
    y_positions = np.arange(len(data), dtype=float)
    fig_height = max(4.0, 0.68 * len(data) + 1.35)
    fig, ax = plt.subplots(figsize=(8.2, fig_height))

    x_values = data["Beta"].to_numpy(dtype=float)
    lower = data["CI_lower"].to_numpy(dtype=float)
    upper = data["CI_upper"].to_numpy(dtype=float)
    xerr = np.vstack([x_values - lower, upper - x_values])

    ax.errorbar(
        x_values,
        y_positions,
        xerr=xerr,
        fmt="o",
        color="black",
        ecolor="black",
        markerfacecolor="black",
        markeredgecolor="black",
        markersize=5.5,
        elinewidth=1.25,
        capsize=3.2,
        capthick=1.25,
        linestyle="none",
        zorder=3,
    )

    ax.axvline(
        0,
        color="#555555",
        linewidth=1.0,
        linestyle=(0, (4, 3)),
        zorder=1,
    )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(data["Combination"].tolist())
    ax.invert_yaxis()
    ax.set_xlabel("Sex coefficient (β) with 95% CI", labelpad=8)
    ax.set_ylabel("")

    # Keep the null-effect reference line visible even when all estimates are positive.
    ci_span = float(upper.max() - lower.min())
    if ci_span <= 0:
        ci_span = 1.0
    x_min = min(0.0, float(lower.min()))
    x_max = float(upper.max())
    ax.set_xlim(x_min - 0.07 * ci_span, x_max + 0.07 * ci_span)
    ax.tick_params(axis="y", length=0, pad=6)
    ax.tick_params(axis="x", direction="out", length=4, width=0.8)
    ax.grid(axis="x", color="#E2E2E2", linewidth=0.6, linestyle=(0, (2, 3)))
    ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.9)
    ax.spines["bottom"].set_linewidth(0.9)

    fig.subplots_adjust(left=0.27, right=0.92, top=0.96, bottom=0.16)
    fig.savefig(
        OUTPUT_FILE,
        dpi=1200,
        format="tiff",
        bbox_inches="tight",
        pad_inches=0.05,
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)

    print(f"Rows plotted: {len(data)}")
    print(data[["Combination", "Beta", "CI_lower", "CI_upper", "q_t"]].to_string(index=False))
    print(f"TIFF (1200 dpi): {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
