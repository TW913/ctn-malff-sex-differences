from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


BASE_DIR = Path(__file__).resolve().parent
GROUP_FILE = BASE_DIR / "motion_metrics分组.csv"
SEX_FILE = BASE_DIR / "motion_metrics男女.csv"
OUTPUT_FILE = BASE_DIR / "Figure_S2_motion_QC.tif"

GROUP_COLUMN = "HC/CTN"
SEX_COLUMN = "sex"
PCT_COLUMN_ALIASES = ("Pct_HighMotion_0_5", "Pct_HighMotion_0.5")
METRIC_COLUMNS = ("MeanFD", "MaxFD", "Pct_HighMotion_0_5")

GROUP_ORDER = ["Patient", "Control"]
SEX_ORDER = ["Male", "Female"]
GROUP_PALETTE = {"Patient": "#E3B23C", "Control": "#4C9A6A"}
SEX_PALETTE = {"Male": "#4472C4", "Female": "#D95F5F"}


def configure_style() -> None:
    """Configure a restrained publication-style theme."""
    font_path = Path(r"C:\Windows\Fonts\times.ttf")
    font_name = "Times New Roman"
    if font_path.exists():
        fm.fontManager.addfont(font_path)
        font_name = fm.FontProperties(fname=font_path).get_name()

    sns.set_theme(
        context="paper",
        style="ticks",
        rc={
            "font.family": font_name,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.linewidth": 0.9,
            "axes.unicode_minus": False,
        },
    )


def load_motion_data(path: Path, require_sex: bool = False) -> pd.DataFrame:
    """Load a CSV and validate all variables used in the figure."""
    data = pd.read_csv(path)

    pct_column = next((c for c in PCT_COLUMN_ALIASES if c in data.columns), None)
    if pct_column is None:
        raise ValueError(
            f"{path.name} is missing the percentage column; expected one of "
            f"{PCT_COLUMN_ALIASES}."
        )
    if pct_column != "Pct_HighMotion_0_5":
        data = data.rename(columns={pct_column: "Pct_HighMotion_0_5"})

    required = ["SubjectID", GROUP_COLUMN, *METRIC_COLUMNS]
    if require_sex:
        required.append(SEX_COLUMN)
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {missing}")

    numeric_columns = [GROUP_COLUMN, *METRIC_COLUMNS]
    if require_sex:
        numeric_columns.append(SEX_COLUMN)
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    if data[required].isna().any().any():
        bad_columns = data[required].columns[data[required].isna().any()].tolist()
        raise ValueError(f"{path.name} has missing/non-numeric values in: {bad_columns}")
    if data["SubjectID"].duplicated().any():
        raise ValueError(f"{path.name} contains duplicate SubjectID values.")
    if not set(data[GROUP_COLUMN]).issubset({0, 1}):
        raise ValueError(f"{path.name} contains unknown HC/CTN codes.")
    if require_sex and not set(data[SEX_COLUMN]).issubset({0, 1}):
        raise ValueError(f"{path.name} contains unknown sex codes.")

    return data


def add_box_and_points(
    ax: plt.Axes,
    data: pd.DataFrame,
    x: str,
    y: str,
    order: list[str],
    palette: dict[str, str],
) -> None:
    """Draw a boxplot without fliers and overlay all observations."""
    sns.boxplot(
        data=data,
        x=x,
        y=y,
        order=order,
        hue=x,
        hue_order=order,
        palette=palette,
        dodge=False,
        legend=False,
        showfliers=False,
        width=0.55,
        saturation=0.78,
        boxprops={"edgecolor": "#222222", "linewidth": 1.0, "alpha": 0.55},
        whiskerprops={"color": "#222222", "linewidth": 1.0},
        capprops={"color": "#222222", "linewidth": 1.0},
        medianprops={"color": "#111111", "linewidth": 1.4},
        ax=ax,
    )
    sns.stripplot(
        data=data,
        x=x,
        y=y,
        order=order,
        hue=x,
        hue_order=order,
        palette=palette,
        dodge=False,
        legend=False,
        alpha=0.70,
        jitter=0.22,
        size=3.0,
        linewidth=0,
        ax=ax,
        zorder=3,
    )

    ax.set_xlabel("")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.6, linestyle=(0, (2, 3)))
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)
    sns.despine(ax=ax)


def main() -> None:
    np.random.seed(20260928)  # Reproducible horizontal jitter.
    configure_style()

    grouped = load_motion_data(GROUP_FILE)
    by_sex = load_motion_data(SEX_FILE, require_sex=True)
    patients = by_sex.loc[by_sex[GROUP_COLUMN].eq(1)].copy()
    if patients.empty:
        raise ValueError("The sex-stratified file contains no patient records.")

    grouped["Diagnostic group"] = grouped[GROUP_COLUMN].map(
        {1: "Patient", 0: "Control"}
    )
    patients["Sex"] = patients[SEX_COLUMN].map({1: "Male", 0: "Female"})

    # Pct_HighMotion is stored as 0-100 percentages in the supplied files.
    pct_max = max(
        grouped["Pct_HighMotion_0_5"].max(),
        patients["Pct_HighMotion_0_5"].max(),
    )
    percentage_scale = pct_max > 1.0
    pct_ylabel = (
        "High-motion volumes (%)"
        if percentage_scale
        else "High-motion volumes (proportion)"
    )

    metric_specs = [
        ("MeanFD", "Mean FD (mm)", "Mean FD"),
        ("MaxFD", "Maximum FD (mm)", "Maximum FD"),
        ("Pct_HighMotion_0_5", pct_ylabel, "High-motion volumes"),
    ]

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(9.6, 6.2),
        constrained_layout=True,
    )
    panel_letters = iter("ABCDEF")

    for column_index, (metric, ylabel, short_title) in enumerate(metric_specs):
        ax = axes[0, column_index]
        add_box_and_points(
            ax,
            grouped,
            x="Diagnostic group",
            y=metric,
            order=GROUP_ORDER,
            palette=GROUP_PALETTE,
        )
        letter = next(panel_letters)
        ax.set_title(f"({letter}) {short_title} by group", loc="left", pad=7)
        ax.set_ylabel(ylabel)
        group_counts = grouped["Diagnostic group"].value_counts()
        ax.set_xticks(
            range(len(GROUP_ORDER)),
            labels=[f"{name}\n(n={group_counts[name]})" for name in GROUP_ORDER],
        )

    for column_index, (metric, ylabel, short_title) in enumerate(metric_specs):
        ax = axes[1, column_index]
        add_box_and_points(
            ax,
            patients,
            x="Sex",
            y=metric,
            order=SEX_ORDER,
            palette=SEX_PALETTE,
        )
        letter = next(panel_letters)
        ax.set_title(f"({letter}) {short_title} by sex", loc="left", pad=7)
        ax.set_ylabel(ylabel)
        sex_counts = patients["Sex"].value_counts()
        ax.set_xticks(
            range(len(SEX_ORDER)),
            labels=[f"{name}\n(n={sex_counts[name]})" for name in SEX_ORDER],
        )

    # Use the same Y-axis range within each metric column for direct comparison.
    for column_index in range(3):
        upper_limit = max(
            axes[0, column_index].get_ylim()[1],
            axes[1, column_index].get_ylim()[1],
        )
        axes[0, column_index].set_ylim(0, upper_limit)
        axes[1, column_index].set_ylim(0, upper_limit)

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

    print(f"Grouped sample: Patient n={(grouped[GROUP_COLUMN] == 1).sum()}, "
          f"Control n={(grouped[GROUP_COLUMN] == 0).sum()}")
    print(f"Patient sex sample: Male n={(patients[SEX_COLUMN] == 1).sum()}, "
          f"Female n={(patients[SEX_COLUMN] == 0).sum()}")
    print(f"TIFF (1200 dpi): {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
