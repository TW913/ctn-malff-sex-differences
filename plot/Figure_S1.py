from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind


BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "参与者信息 .xlsx"
OUTPUT_TIF = BASE_DIR / "性别_疼痛评分_散点图.tif"

# 编码根据项目元数据：1 = male，0 = female。
GROUPS = [
    {"code": 1, "label": "Male", "color": "#2F6DB3"},
    {"code": 0, "label": "Female", "color": "#D64B4B"},
]


def set_plot_font() -> None:
    """将图内所有文字统一为 12 pt Times New Roman。"""
    font_path = Path(r"C:\Windows\Fonts\times.ttf")
    if not font_path.exists():
        raise FileNotFoundError("未找到 Times New Roman 字体文件。")
    fm.fontManager.addfont(font_path)
    font_name = fm.FontProperties(fname=font_path).get_name()
    plt.rcParams.update(
        {
            "font.family": font_name,
            "font.size": 12,
            "axes.labelsize": 12,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
        }
    )
    plt.rcParams["axes.unicode_minus"] = False


def load_data() -> pd.DataFrame:
    """读取并校验绘图所需的性别和疼痛评分数据。"""
    data = pd.read_excel(DATA_FILE, sheet_name="总表")
    required_columns = ["性别", "疼痛评分"]
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(f"工作表缺少列：{missing_columns}")

    plot_data = data[required_columns].copy()
    plot_data["性别"] = pd.to_numeric(plot_data["性别"], errors="coerce")
    plot_data["疼痛评分"] = pd.to_numeric(
        plot_data["疼痛评分"], errors="coerce"
    )
    plot_data = plot_data.dropna(subset=required_columns)

    unexpected_codes = sorted(set(plot_data["性别"]) - {0, 1})
    if unexpected_codes:
        raise ValueError(f"发现未知性别编码：{unexpected_codes}")
    if plot_data.empty:
        raise ValueError("没有可用的性别与疼痛评分数据。")

    return plot_data


def main() -> None:
    set_plot_font()
    data = load_data()

    male_scores = data.loc[data["性别"] == 1, "疼痛评分"]
    female_scores = data.loc[data["性别"] == 0, "疼痛评分"]
    test = ttest_ind(male_scores, female_scores, equal_var=False)

    fig, ax = plt.subplots(figsize=(5.6, 5.8), constrained_layout=True)
    rng = np.random.default_rng(20260928)

    ymax = float(data["疼痛评分"].max())
    bracket_y = ymax + 0.62
    bracket_height = 0.24
    y_upper = bracket_y + 1.05

    for x_pos, group in enumerate(GROUPS):
        scores = data.loc[
            data["性别"] == group["code"], "疼痛评分"
        ].to_numpy(dtype=float)
        mean = float(np.mean(scores))
        sd = float(np.std(scores, ddof=1))
        jitter = rng.uniform(-0.15, 0.15, size=len(scores))

        ax.scatter(
            np.full(len(scores), x_pos) + jitter,
            scores,
            s=58,
            color=group["color"],
            alpha=0.85,
            edgecolor="white",
            linewidth=0.7,
            zorder=3,
        )

        # 黑色横线表示均值，误差线表示均值 ± 标准差。
        ax.errorbar(
            x_pos,
            mean,
            yerr=sd,
            fmt="none",
            color="#202020",
            ecolor="#202020",
            elinewidth=1.8,
            capsize=7,
            capthick=1.8,
            zorder=5,
        )
        ax.hlines(
            mean,
            x_pos - 0.22,
            x_pos + 0.22,
            color="#202020",
            linewidth=2.0,
            zorder=4,
        )
        ax.text(
            x_pos,
            0.36,
            f"Mean ± SD\n{mean:.2f} ± {sd:.2f}",
            ha="center",
            va="center",
            fontsize=12,
            fontweight="semibold",
            color="#303030",
        )

    # 组间显著性横线及三颗星（p < 0.001）。
    ax.plot(
        [0, 0, 1, 1],
        [bracket_y, bracket_y + bracket_height, bracket_y + bracket_height, bracket_y],
        color="#202020",
        linewidth=1.6,
        clip_on=False,
    )
    ax.text(
        0.5,
        bracket_y + bracket_height + 0.10,
        "***",
        ha="center",
        va="bottom",
        fontsize=12,
        fontweight="bold",
        color="#202020",
    )

    ax.set_ylabel("Pain severity", fontsize=12)
    ax.set_xlabel("")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(
        [
            f"Male\n(n={len(male_scores)})",
            f"Female\n(n={len(female_scores)})",
        ],
        fontsize=12,
    )
    ax.set_xlim(-0.48, 1.48)
    ax.set_ylim(0, y_upper)
    ax.set_yticks(np.arange(0, ymax + 0.1, 2))
    ax.tick_params(axis="y", labelsize=12, length=4, width=1)
    ax.tick_params(axis="x", length=0, pad=8)
    ax.yaxis.grid(True, color="#D9D9D9", linewidth=0.8, linestyle=(0, (2, 3)))
    ax.xaxis.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.1)
    ax.spines["bottom"].set_linewidth(1.1)
    ax.set_axisbelow(True)

    fig.savefig(
        OUTPUT_TIF,
        dpi=1200,
        format="tiff",
        bbox_inches="tight",
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)

    print(f"Welch t = {test.statistic:.4f}, p = {test.pvalue:.6g}")
    for group in GROUPS:
        scores = data.loc[data["性别"] == group["code"], "疼痛评分"]
        print(
            f"{group['label']}: n={len(scores)}, "
            f"均值={scores.mean():.2f}, 标准差={scores.std(ddof=1):.2f}"
        )
    print(f"TIFF (1200 dpi): {OUTPUT_TIF}")


if __name__ == "__main__":
    main()
