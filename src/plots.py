"""独立绘图：只读取已有计算结果，不导入或调用优化求解器。

集中保留边界、代表周调度、发生年份 NPV 分位带、三层 NPV 热力面、
重优化敏感性三栏图；论文主图沿用既有绘图函数及配色。
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import csv
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.collections import PolyCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np
from scipy.spatial import cKDTree
from src import config as cfg

FIG = cfg.OUTPUT_ROOT / "figures"
WANYUAN = 10_000.0
YEARS = (1, 6, 11)
HALF_WIDTH = 2.30
PLANE_HEIGHT = 1.10
LAYER_STEP = 1.20
SLATE = "#344352"
DARK = "#263445"
GROUPS = ["光伏投资成本", "储能投资成本", "合同价格系数", "需量分成", "分时电价", "折现率", "上网电价"]
plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
                     "axes.unicode_minus": False, "svg.fonttype": "none"})

DEEP_BLUE = "#3951A2"
LIGHT_BLUE = "#72AACF"
PALE_BLUE = "#CAE8F2"
PALE_YELLOW = "#FEFBBA"
ORANGE = "#FDB96B"
RED_ORANGE = "#EC5D3B"
DEEP_RED = "#A80326"
PALETTE = (DEEP_BLUE, LIGHT_BLUE, PALE_BLUE, PALE_YELLOW, ORANGE, RED_ORANGE, DEEP_RED)

GRID = PALE_BLUE
_FS = 7.5
_LW = 1.2

# 局部边界加密结果：PV-only 首次满足最低消纳约束的负荷拐点。
REFINED_KINK_ALPHA = 0.127638519129
# α≈0.128 时 γ=0.7444/0.7445 的储能价值差异号相反，取局部接点中值。
REFINED_KINK_GAMMA = 0.74445


def _style(ax):
    ax.tick_params(labelsize=_FS - 0.5)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["bottom"].set_linewidth(0.6)
    ax.grid(True, color=GRID, linewidth=0.4, alpha=0.6)
    ax.set_axisbelow(True)
    return ax


def _figure(*, width: float = 3.5, height: float = 2.9):
    fig, ax = plt.subplots(figsize=(width, height), dpi=150)
    return fig, _style(ax)


def _save(fig, name: str, output: Path = FIG):
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def state_boundaries_plot(path=None, *, output: Path = FIG, allow_legacy: bool = False):
    """绘制三状态区域图：投资进入线和储能升级线独立平滑连接。

    只接受新边界扫描正式结果，不从旧断点或手工估计点补数据。低负荷
    下若 PV-only 状态尚未物理可用，进入线以上直接填充为光伏+储能；
    两状态均可用后，储能升级线把仅光伏和光伏+储能分开。
    """
    path = Path(path or cfg.OUTPUT_ROOT / "f2_e1" / "e1_state_boundaries.json")
    if not path.exists():
        return
    result = json.loads(path.read_text(encoding="utf-8"))
    expected_version = "vp-vb-value-priority-v5-gamma-tou-actual-investor-npv"
    if not allow_legacy and result.get("classification_version") != expected_version:
        return

    def threshold_points(curve: dict) -> tuple[np.ndarray, np.ndarray]:
        rows = [
            point
            for point in curve.get("points", [])
            if point.get("status") == "threshold"
            and point.get("threshold") is not None
        ]
        pairs = sorted(
            (float(point["alpha"]), float(point["threshold"])) for point in rows
        )
        if not pairs:
            return np.asarray([], dtype=float), np.asarray([], dtype=float)
        unique: dict[float, float] = {}
        for alpha, gamma in pairs:
            unique[alpha] = gamma
        return (
            np.asarray(list(unique.keys()), dtype=float),
            np.asarray(list(unique.values()), dtype=float),
        )

    def smooth(x: np.ndarray, y: np.ndarray, dense: np.ndarray) -> np.ndarray:
        if len(x) == 1:
            return np.full_like(dense, y[0], dtype=float)
        try:
            from scipy.interpolate import PchipInterpolator

            if len(x) >= 3:
                return PchipInterpolator(x, y)(dense)
        except Exception:  # noqa: BLE001
            pass
        return np.interp(dense, x, y)

    entry_x, entry_y = threshold_points(result.get("pv_only_entry", {}))
    alpha_min = result.get("alpha_min_gamma_1", {})
    if isinstance(alpha_min, dict) and alpha_min.get("threshold") is not None:
        if alpha_min.get("status") in {"threshold", "all_feasible"}:
            anchor = (float(alpha_min["threshold"]), 1.0)
            if not any(abs(alpha - anchor[0]) <= 1e-8 for alpha in entry_x):
                entry_x = np.append(entry_x, anchor[0])
                entry_y = np.append(entry_y, anchor[1])
                order = np.argsort(entry_x)
                entry_x, entry_y = entry_x[order], entry_y[order]

    upgrade_x, upgrade_y = threshold_points(result.get("pv_battery_entry", {}))
    if len(entry_x) < 2 or len(upgrade_x) < 2:
        raise RuntimeError(
            "正式边界结果不足：投资进入线和储能升级线均至少需要两个有效散点。"
        )

    gamma_range = result.get("gamma_range", [0.0, 1.0])
    # Keep the full discount-factor domain visible in the figure.
    gamma_min = 0.0
    gamma_max = 1.0
    if len(gamma_range) >= 2:
        gamma_min = max(0.0, float(gamma_range[0]))
        gamma_max = min(1.0, float(gamma_range[1]))
    if gamma_max <= gamma_min:
        raise RuntimeError("invalid gamma_range for state-boundary figure")
    dense = np.linspace(0.0, 1.0, 800)

    entry_curve = np.full_like(dense, gamma_max)
    entry_mask = dense >= entry_x[0]
    entry_curve[entry_mask] = np.clip(
        smooth(entry_x, entry_y, dense[entry_mask]), gamma_min, gamma_max
    )

    # There are two disconnected orange segments. The first is a visual
    # companion for the direct no-investment -> PV+battery branch: it starts
    # at the gamma=1 investment anchor and ends at the refined load kink. The
    # second is anchored at the same refined alpha and then follows the
    # formally scanned PV-only -> PV+battery boundary.
    kink_x = REFINED_KINK_ALPHA
    kink_y = float(np.interp(kink_x, dense, entry_curve))
    upgrade_plot_x = upgrade_x.copy()
    upgrade_plot_y = upgrade_y.copy()
    if kink_x < upgrade_plot_x[0] - 1e-12:
        upgrade_plot_x = np.insert(upgrade_plot_x, 0, kink_x)
        upgrade_plot_y = np.insert(upgrade_plot_y, 0, REFINED_KINK_GAMMA)

    # Keep the blue boundary fixed and retain a clear parallel gap for almost
    # the whole direct-jump branch. Only the final offset-width end cap returns
    # to the blue kink, keeping the kink markers vertically aligned without a
    # long visually merged tail.
    left_branch_mask = (dense >= entry_x[0]) & (dense <= kink_x)
    left_blue_x = dense[left_branch_mask]
    left_upgrade_y = entry_curve[left_branch_mask]
    alpha_shift = 0.006
    branch_span = max(kink_x - float(entry_x[0]), np.finfo(float).eps)
    taper_width = min(alpha_shift, branch_span)
    taper_start = max(0.0, 1.0 - taper_width / branch_span)
    left_t = np.clip(
        (left_blue_x - float(entry_x[0])) / branch_span,
        0.0,
        1.0,
    )
    left_offset = np.where(
        left_t <= taper_start,
        alpha_shift,
        alpha_shift * (1.0 - left_t) / (1.0 - taper_start),
    )
    left_upgrade_x = left_blue_x + left_offset
    left_marker_mask = entry_x <= kink_x
    left_marker_base_x = entry_x[left_marker_mask]
    left_marker_base_y = entry_y[left_marker_mask]
    if not np.any(np.isclose(left_marker_base_x, kink_x, atol=1e-10)):
        left_marker_base_x = np.append(left_marker_base_x, kink_x)
        left_marker_base_y = np.append(left_marker_base_y, kink_y)
    marker_t = np.clip(
        (left_marker_base_x - float(entry_x[0])) / branch_span,
        0.0,
        1.0,
    )
    marker_offset = np.where(
        marker_t <= taper_start,
        alpha_shift,
        alpha_shift * (1.0 - marker_t) / (1.0 - taper_start),
    )
    left_orange_marker_x = left_marker_base_x + marker_offset
    left_orange_marker_y = left_marker_base_y
    upgrade_curve = np.full_like(dense, gamma_max)
    upgrade_mask = dense >= upgrade_plot_x[0]
    upgrade_curve[upgrade_mask] = np.clip(
        smooth(upgrade_plot_x, upgrade_plot_y, dense[upgrade_mask]),
        gamma_min,
        gamma_max,
    )

    fig, ax = _figure(width=6.2, height=4.1)
    ax.grid(False)
    ax.axvline(
        kink_x,
        color="#6B6B6B",
        linestyle="--",
        linewidth=0.8,
        alpha=0.75,
        zorder=1,
    )
    ax.fill_between(
        dense, gamma_min, entry_curve, color=PALE_YELLOW, alpha=0.92, zorder=0
    )

    # Only the right-hand branch contains a PV-only region.
    upgrade_valid = (
        (dense >= kink_x) & (upgrade_curve > entry_curve + 1e-9)
    )
    ax.fill_between(
        dense[upgrade_valid],
        entry_curve[upgrade_valid],
        upgrade_curve[upgrade_valid],
        color=PALE_BLUE,
        alpha=0.92,
        zorder=0,
    )
    battery_lower = np.where(upgrade_valid, upgrade_curve, entry_curve)
    ax.fill_between(
        dense, battery_lower, gamma_max, color=LIGHT_BLUE, alpha=0.92, zorder=0
    )

    ax.plot(
        dense[entry_mask],
        entry_curve[entry_mask],
        color=DEEP_BLUE,
        lw=2.0,
        label="Investment entry",
        zorder=7,
    )
    ax.plot(
        left_upgrade_x,
        left_upgrade_y,
        color=RED_ORANGE,
        lw=2.0,
        label="Battery boundary (parallel direct-jump branch)",
        zorder=5,
    )
    ax.plot(
        dense[upgrade_mask],
        upgrade_curve[upgrade_mask],
        color=RED_ORANGE,
        lw=2.0,
        label="Battery upgrade",
        zorder=5,
    )
    ax.scatter(entry_x, entry_y, s=10, color=DEEP_BLUE, zorder=8)
    ax.scatter(
        left_orange_marker_x,
        left_orange_marker_y,
        s=10,
        color=RED_ORANGE,
        marker="o",
        zorder=6,
    )
    ax.scatter(upgrade_plot_x, upgrade_plot_y, s=10, color=RED_ORANGE, zorder=6)

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(gamma_min, gamma_max)
    ax.set_xlabel("Load factor $\\alpha$", fontsize=_FS)
    ax.set_ylabel("Price factor $\\gamma$", fontsize=_FS)
    ax.legend(fontsize=_FS - 1, frameon=False, loc="best")
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / "fig_state_boundaries.png", dpi=300, bbox_inches="tight")
    fig.savefig(output / "fig_state_boundaries.svg", bbox_inches="tight")

    # 无文字标注版本：保留坐标轴、刻度线、网格和边界/填充，隐藏所有文字。
    ax.set_xlabel("")
    ax.set_ylabel("")
    if ax.legend_ is not None:
        ax.legend_.remove()
    ax.tick_params(
        axis="both",
        which="both",
        length=3.0,
        labelbottom=False,
        labelleft=False,
        labeltop=False,
        labelright=False,
    )
    fig.savefig(
        output / "fig_state_boundaries_no_labels.svg",
        bbox_inches="tight",
        transparent=True,
    )
    plt.close(fig)


def weekly_dispatch(path: Path = cfg.BASELINE_OUTPUT / "baseline_result.json", week=20, *, output: Path = FIG):
    """代表周调度：供能堆叠 + 负荷/光伏线 + 上网负向条 + SOC 右轴。"""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    r = payload["result"]
    if r["status"] != "optimal":
        return
    if not 1 <= week <= 52:
        raise ValueError("week 必须在 1 到 52 之间")
    d = {key: np.asarray(values, dtype=float) for key, values in r["dispatch"].items()}
    data = {key: np.asarray(payload[key], dtype=float) for key in ("load", "pv_profile", "tou")}
    s, e = (week - 1) * 168, week * 168
    x = np.arange(s, e)
    fig, ax = plt.subplots(figsize=(7.0, 2.9), dpi=150)
    _style(ax)
    ax.stackplot(
        x,
        d["pv_to_user"][s:e],
        d["ess_to_user"][s:e],
        d["grid_to_user"][s:e],
        labels=["PV direct", "ESS discharge", "Grid purchase"],
        colors=[ORANGE, LIGHT_BLUE, PALE_BLUE],
        alpha=0.85,
    )
    ax.plot(x, data["load"][s:e], color="black", lw=0.9, ls="--", label="Load")
    ax.plot(
        x, r["pv_kw"] * data["pv_profile"][s:e], color=RED_ORANGE, lw=0.8, label="PV output"
    )
    ax.bar(
        x, -d["export"][s:e], width=1.0, color=PALE_YELLOW, alpha=0.9, label="Export"
    )
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xlabel("Hour of year", fontsize=_FS)
    ax.set_ylabel("Power (kW)", fontsize=_FS)
    ax.legend(fontsize=_FS - 1, frameon=False, ncol=2, loc="upper left")
    ax2 = ax.twinx()
    ax2.plot(x, d["soc"][s:e], color=DEEP_BLUE, lw=0.9, ls=":", label="SOC")
    ax2.set_ylabel("SOC (kWh)", fontsize=_FS, color=DEEP_BLUE)
    ax2.tick_params(labelsize=_FS - 1)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_linewidth(0.6)
    _save(fig, "fig_weekly_dispatch", output)




def _prepare_svg_overlay(fig: mpl.figure.Figure) -> None:
    """Hide labels and text for the transparent PPT overlay SVG only."""
    for ax in fig.axes:
        ax.set_xlabel("")
        ax.set_ylabel("")
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
        ax.tick_params(
            axis="both",
            which="both",
            labelbottom=False,
            labelleft=False,
            labeltop=False,
            labelright=False,
        )
        for text in ax.texts:
            text.set_visible(False)
    for text in fig.texts:
        text.set_visible(False)


def load_onset_data(source: Path) -> tuple[list[dict[str, float]], int, float]:
    summary = json.loads((source / "run_summary.json").read_text(encoding="utf-8"))
    if summary["status"] != "complete" or not all(summary["checks"].values()):
        raise RuntimeError("主样本尚未完成或未通过审计")
    if not np.isclose(
        float(summary.get("locked_export_max_kw", np.nan)),
        cfg.LOCKED_EXPORT_MAX_KW,
    ):
        raise RuntimeError("主样本未使用锁定压力状态的 1000 kW 并网容量限制")
    n = int(summary["scenario_count"])
    if n != 1000 or summary["result_rows"] != 15 * n:
        raise RuntimeError("主样本应为 1000 个场景、15 个发生年份")

    with (source / "timing_results.csv").open(encoding="utf-8-sig", newline="") as handle:
        raw = list(csv.DictReader(handle))
    if len(raw) != 15 * n:
        raise RuntimeError("明细行数不符")
    pairs = Counter((int(row["scenario_id"]), int(row["onset_year"])) for row in raw)
    if len(pairs) != 15 * n or any(count != 1 for count in pairs.values()):
        raise RuntimeError("场景与发生年份组合不完整或重复")

    by_year = {year: [] for year in range(1, 16)}
    for row in raw:
        by_year[int(row["onset_year"])].append(row)
    expected_years = {int(item["onset_year"]) for item in summary["summary"]}
    if expected_years != set(range(1, 16)):
        raise RuntimeError("汇总中缺少发生年份")
    sample_sizes = {len(items) for items in by_year.values()}
    if len(sample_sizes) != 1 or not sample_sizes:
        raise RuntimeError("各发生年份的样本数不一致")
    original_n = next(iter(sample_sizes))
    if original_n != n:
        raise RuntimeError("每个发生年份必须使用全部正式 LHS 场景")
    results: list[dict[str, float]] = []
    for year, items in by_year.items():
        if len(items) != original_n:
            raise RuntimeError(f"第 {year} 年原始场景数不符")
        gamma = np.array([float(row["gamma"]) for row in items])
        if np.any(gamma < 0.4 - 1e-10) or np.any(gamma > 1.0 + 1e-10):
            raise RuntimeError("价格系数超出正式 LHS 范围")
        npv = np.array([float(row["npv_actual_investor"]) for row in items])
        if not np.isfinite(npv).all():
            raise RuntimeError("明细含非有限值")
        quantiles = np.quantile(npv, [0.05, 0.25, 0.50, 0.75, 0.95])
        results.append({
            "onset_year": year,
            "n": len(items),
            "excluded_n": 0,
            "npv_q05_yuan": float(quantiles[0]),
            "npv_q25_yuan": float(quantiles[1]),
            "npv_q50_yuan": float(quantiles[2]),
            "npv_q75_yuan": float(quantiles[3]),
            "npv_q95_yuan": float(quantiles[4]),
        })
    baseline_npv_wanyuan = float(summary["baseline_npv"]) / WANYUAN
    return results, original_n, baseline_npv_wanyuan


def draw_onset_npv(rows: list[dict[str, float]], n: int, baseline_npv_wanyuan: float, *, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    table = output / "figure_source_data.csv"
    with table.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial", "DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "axes.linewidth": 0.75,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "savefig.facecolor": "white",
    })
    years = np.array([row["onset_year"] for row in rows])
    q = {p: np.array([row[f"npv_q{p:02d}_yuan"] / WANYUAN for row in rows]) for p in (5, 25, 50, 75, 95)}
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.fill_between(years, q[5], q[95], color=PALE_BLUE, label="P5–P95", zorder=1)
    ax.fill_between(years, q[25], q[75], color=LIGHT_BLUE, alpha=0.60, label="P25–P75", zorder=2)
    ax.plot(years, q[50], color=DEEP_BLUE, lw=1.9, marker="o", ms=2.9, label="中位数", zorder=3)
    ax.axhline(
        baseline_npv_wanyuan,
        color="#6b7280",
        lw=1.1,
        ls=(0, (4, 2)),
        label=f"基准 NPV={baseline_npv_wanyuan:.2f} 万元",
        zorder=2.5,
    )
    ymin = float(np.nanmin(q[5]))
    ymax = float(np.nanmax(q[95]))
    pad = max(10.0, 0.06 * (ymax - ymin))
    lower = np.floor((ymin - pad) / 10.0) * 10.0
    upper = np.ceil((ymax + pad) / 10.0) * 10.0
    ax.set(xlim=(1, 15), ylim=(lower, upper), xlabel="压力发生年份 τ", ylabel="投资方实际 NPV（万元）")
    ax.set_xticks([1, 3, 6, 9, 12, 15])
    ax.set_yticks(np.arange(np.ceil(lower / 50.0) * 50.0, upper + 1.0, 50.0))
    ax.grid(False)
    ax.legend(loc="upper right", ncol=2, fontsize=6.5, handlelength=1.5, columnspacing=0.8)
    fig.text(0.02, 0.015,
             f"每年使用全部正式 LHS 场景 n={n}；γ=0.4–1.0，不筛选 NPV 损失；容量固定",
             fontsize=6.6, color="black")
    fig.subplots_adjust(left=0.12, right=0.985, top=0.95, bottom=0.19)
    stem = output / "fig3_onset_actual_npv_full_lhs"
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".tiff"), dpi=600)
    _prepare_svg_overlay(fig)
    fig.savefig(stem.with_suffix(".svg"), transparent=True)
    plt.close(fig)

    (output / "figure_note.md").write_text(
        "# 图 3｜正式 LHS 全域的锁定容量实际 NPV\n\n"
        "使用正式 α=0.25–1.0、γ=0.4–1.0 LHS 的全部 1000 个场景，"
        "逐发生年份展示实际 NPV P5–P95、P25–P75 与中位数，不作条件筛选。"
        f"唯一虚线为基准规划 NPV={baseline_npv_wanyuan:.2f} 万元。"
        f"每个发生年份使用 {n} 个正式 LHS 压力场景。"
        "分位带是研究抽样域内的经验分布，不是现实发生概率或置信区间。"
        "基准状态保留规划期 70% 消纳率要求，压力状态不强制消纳率，"
        "并设置 1000 kW 并网容量上限；光伏逐小时优先直供负荷，"
        "超过上限的余电弃光。"
        "第1、6、11年三层热力图另行展示负荷与合同价格共同作用下的实际NPV空间分布；"
        "两图使用同一正式LHS主实验。来源：固定容量主样本的15000条发生年份结果。\n",
        encoding="utf-8",
    )


def shared_color_scale(source: Path):
    summary = json.loads((source / "run_summary.json").read_text(encoding="utf-8"))
    if summary.get("status") != "complete" or summary.get("scenario_count") != 1000:
        raise RuntimeError("正式 LHS 主实验未完成")
    values = []
    counts = {year: 0 for year in YEARS}
    with (source / "timing_results.csv").open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            year = int(row["onset_year"])
            if year in counts:
                values.append(float(row["npv_actual_investor"]) / 1e4)
                counts[year] += 1
    if any(n != 1000 for n in counts.values()) or not np.isfinite(values).all():
        raise RuntimeError("三年的正式 LHS 实际 NPV 样本不完整")
    low, high = min(values), max(values)
    baseline = float(summary["baseline_npv"]) / 1e4
    if not low < baseline < high:
        raise RuntimeError("基准 NPV 必须位于共用色标范围之内")
    cmap = LinearSegmentedColormap.from_list("lhs_npv_blue_low_red_high", PALETTE, N=256)
    norm = Normalize(vmin=low, vmax=high)
    return cmap, norm, {"min": low, "baseline": baseline, "max": high,
                        "mapping": "linear"}


def load_surface_data(source: Path) -> tuple[dict[int, list[dict]], dict]:
    summary = json.loads((source / "run_summary.json").read_text(encoding="utf-8"))
    if (summary.get("status") != "complete" or summary.get("scenario_count") != 1000
            or summary.get("result_rows") != 15000 or not all(summary.get("checks", {}).values())):
        raise RuntimeError("正式 LHS 未完成或没有通过全部审计")
    rows = {year: [] for year in YEARS}
    with (source / "timing_results.csv").open(encoding="utf-8-sig", newline="") as handle:
        for raw in csv.DictReader(handle):
            year = int(raw["onset_year"])
            if year not in rows:
                continue
            item = {
                "scenario_id": int(raw["scenario_id"]),
                "onset_year": year,
                "alpha": float(raw["alpha"]),
                "gamma": float(raw["gamma"]),
                "npv_yuan": float(raw["npv_actual_investor"]),
                "user_constraint_ok": int(raw["user_constraint_ok"]),
            }
            rows[year].append(item)
    ids = None
    for year, group in rows.items():
        group.sort(key=lambda row: row["scenario_id"])
        if len(group) != 1000 or len({row["scenario_id"] for row in group}) != 1000:
            raise RuntimeError(f"第 {year} 年不是完整的 1000 个正式 LHS 场景")
        if any(row["user_constraint_ok"] != 1 for row in group):
            raise RuntimeError(f"第 {year} 年用户约束存在未通过情景")
        xyz = np.array([(row["alpha"], row["gamma"], row["npv_yuan"]) for row in group])
        if not np.isfinite(xyz).all():
            raise RuntimeError(f"第 {year} 年存在非有限值")
        current_ids = [row["scenario_id"] for row in group]
        if ids is None:
            ids = current_ids
        elif ids != current_ids:
            raise RuntimeError("三个发生年份的情景 ID 不一致")
    for a, b in ((1, 6), (6, 11)):
        xy_a = np.array([(row["alpha"], row["gamma"]) for row in rows[a]])
        xy_b = np.array([(row["alpha"], row["gamma"]) for row in rows[b]])
        if not np.allclose(xy_a, xy_b, atol=1e-12, rtol=0):
            raise RuntimeError("三个发生年份未使用同一组 LHS 负荷与价格点")
    return rows, summary


def interpolate_surface(group: list[dict], mesh_alpha: np.ndarray,
                        mesh_gamma: np.ndarray) -> tuple[np.ndarray, int]:
    alpha = np.array([row["alpha"] for row in group])
    gamma = np.array([row["gamma"] for row in group])
    npv = np.array([row["npv_yuan"] for row in group]) / WANYUAN
    tri = mtri.Triangulation(alpha, gamma)
    linear = np.ma.asarray(mtri.LinearTriInterpolator(tri, npv)(mesh_alpha, mesh_gamma))
    mask = np.ma.getmaskarray(linear) | ~np.isfinite(np.asarray(linear.filled(np.nan)))
    tree = cKDTree(np.column_stack((alpha, gamma)))
    _, nearest = tree.query(np.column_stack((mesh_alpha.ravel(), mesh_gamma.ravel())))
    nearest_values = npv[nearest].reshape(mesh_alpha.shape)
    filled = np.where(mask, nearest_values, linear.filled(np.nan))
    if not np.isfinite(filled).all():
        raise RuntimeError("插值表面出现非有限值")
    return filled, int(mask.sum())


def projected(u: np.ndarray, v: np.ndarray, bottom: float) -> tuple[np.ndarray, np.ndarray]:
    """将实际 α、γ 输入映射为叠层热力面的显示坐标。"""
    return HALF_WIDTH * (u - v), bottom + 0.5 * PLANE_HEIGHT * (u + v)


def plane_mesh(group: list[dict], bottom: float, cmap, norm) -> tuple[PolyCollection, int]:
    alpha_axis = np.linspace(0.25, 1.0, 61)
    gamma_axis = np.linspace(0.4, 1.0, 61)
    aa, gg = np.meshgrid(alpha_axis, gamma_axis)
    surface, filled_cells = interpolate_surface(group, aa, gg)
    u = (aa - 0.25) / 0.75
    v = (gg - 0.4) / 0.6
    xx, yy = projected(u, v, bottom)
    quads = np.stack((
        np.stack((xx[:-1, :-1], yy[:-1, :-1]), axis=-1),
        np.stack((xx[:-1, 1:], yy[:-1, 1:]), axis=-1),
        np.stack((xx[1:, 1:], yy[1:, 1:]), axis=-1),
        np.stack((xx[1:, :-1], yy[1:, :-1]), axis=-1),
    ), axis=2)
    values = 0.25 * (surface[:-1, :-1] + surface[:-1, 1:]
                     + surface[1:, 1:] + surface[1:, :-1])
    collection = PolyCollection(
        quads.reshape(-1, 4, 2), facecolors=cmap(norm(values.ravel())),
        edgecolors="none", linewidths=0, antialiaseds=False,
        rasterized=True,
    )
    return collection, filled_cells


def _outline(ax, bottom: float) -> None:
    u = np.array([0.0, 1.0, 1.0, 0.0, 0.0])
    v = np.array([0.0, 0.0, 1.0, 1.0, 0.0])
    x, y = projected(u, v, bottom)
    ax.plot(x, y, color="#637487", lw=0.8, alpha=0.85, zorder=4)


def _hide_surface_text(fig: mpl.figure.Figure) -> None:
    fig.patch.set_alpha(0.0)
    for ax in fig.axes:
        ax.set_facecolor("none")
        ax.tick_params(labelleft=False, labelright=False, labeltop=False, labelbottom=False)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title("")
        for label in ax.texts:
            label.set_visible(False)
    for label in fig.texts:
        label.set_visible(False)


def draw_npv_surfaces(groups: dict[int, list[dict]], summary: dict, *, source: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    with (output / "figure_source_data.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        rows = [row for year in YEARS for row in groups[year]]
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    cmap, norm, scale = shared_color_scale(source)
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial", "DejaVu Sans"],
        "svg.fonttype": "none", "pdf.fonttype": 42,
        "font.size": 9,
    })
    fig = plt.figure(figsize=(8.1, 5.45), facecolor="white")
    ax = fig.add_axes((0.06, 0.10, 0.76, 0.78))
    ax.set_aspect("equal")
    ax.set_axis_off()
    offsets = {year: index * LAYER_STEP for index, year in enumerate(YEARS)}
    top = offsets[YEARS[-1]] + PLANE_HEIGHT

    fill_counts = {}
    for year in YEARS:
        bottom = offsets[year]
        collection, count = plane_mesh(groups[year], bottom, cmap, norm)
        collection.set_zorder(2)
        ax.add_collection(collection)
        fill_counts[str(year)] = count
        ax.text(-HALF_WIDTH - 0.15, bottom + 0.5 * PLANE_HEIGHT,
                str(year), ha="right", va="center", fontsize=8.5, color=SLATE)

    # 仅强调一组坐标轴：底层共同起点向左右延伸的两条输入轴，
    # 以及从该点向上的年份轴；不在每个热力面重复画 x/y 轴。
    axis_color = "#111111"
    axis_width = 0.5
    ax.plot([0, -HALF_WIDTH], [0, 0.5 * PLANE_HEIGHT],
            color=axis_color, lw=axis_width, zorder=6)
    ax.plot([0, HALF_WIDTH], [0, 0.5 * PLANE_HEIGHT],
            color=axis_color, lw=axis_width, zorder=6)
    year_ticks = [offsets[year] + 0.5 * PLANE_HEIGHT for year in YEARS]
    # 年份轴直接落在各热力面的左右端点，不再悬空。
    for year_axis_x in (-HALF_WIDTH, HALF_WIDTH):
        ax.plot([year_axis_x, year_axis_x], [year_ticks[0], year_ticks[-1]],
                color=axis_color, lw=axis_width, zorder=6)
        if year_axis_x < 0:
            for tick in year_ticks:
                ax.plot([year_axis_x, year_axis_x - 0.055], [tick, tick],
                        color=axis_color, lw=axis_width, zorder=6)

    # 底面直接标注实际 α、γ 范围；不显示归一化坐标或人为的 0 起点。
    axis_specs = (
        (-1, ((0.40, "0.4"), (0.60, "0.6"), (0.80, "0.8"), (1.00, "1.0")),
         "γ：EMC合同价格系数", -13),
        (1, ((0.25, "0.25"), (0.50, "0.50"), (0.75, "0.75"), (1.00, "1.0")),
         "α：负荷保留率", 13),
    )
    for side, value_labels, axis_label, rotation in axis_specs:
        values = tuple(value for value, _ in value_labels)
        for index, (value, label) in enumerate(value_labels):
            t = (value - values[0]) / (values[-1] - values[0])
            x, y = side * HALF_WIDTH * t, 0.5 * PLANE_HEIGHT * t
            # 刻度沿外侧、平行于另一条底面斜轴。
            dx = side * 0.065
            dy = -0.5 * PLANE_HEIGHT * (0.065 / HALF_WIDTH)
            ax.plot([x, x + dx], [y, y + dy],
                    color=axis_color, lw=axis_width, zorder=7)
            label_offset = 0.18 if t == 0 else 0.055
            ax.text(x + (label_offset if side > 0 else -label_offset), y - 0.13,
                    label, ha="center", va="top",
                    fontsize=7.3, color=SLATE)
        ax.text(side * HALF_WIDTH * 0.73, -0.19, axis_label, rotation=rotation,
                ha="center", va="top", fontsize=8.2, color=SLATE)
    ax.text(-HALF_WIDTH - 0.61, 0.5 * top, "压力发生年份 τ", rotation=90,
            ha="center", va="center", fontsize=8.5, color=SLATE)
    ax.set_xlim(-3.06, 2.75)
    ax.set_ylim(-0.68, top + 0.20)

    scalar = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    scalar.set_array([])
    cax = fig.add_axes((0.84, 0.27, 0.025, 0.47))
    colorbar = fig.colorbar(scalar, cax=cax, ticks=np.arange(-100, 401, 100))
    colorbar.set_label("投资方实际 NPV（万元）", labelpad=9)
    colorbar.ax.tick_params(labelsize=7.5, length=3)
    fig.suptitle("第 1、6、11 年实际 NPV 叠层热力图", x=0.06, y=0.97,
                 ha="left", fontsize=11.5, color=SLATE)
    fig.text(0.07, 0.042,
             "坐标轴直接使用实际范围：α=0.25–1.00，γ=0.40–1.00；三层共用线性色标。",
             fontsize=7.4, color="#566272")
    stem = output / "fig6_lhs_npv_isometric_linear_years_1_6_11"
    fig.savefig(stem.with_suffix(".png"), dpi=320, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), dpi=600, bbox_inches="tight")
    _hide_surface_text(fig)
    fig.savefig(stem.with_suffix(".svg"), transparent=True, facecolor="none",
                edgecolor="none", dpi=600, bbox_inches="tight")
    plt.close(fig)

    metadata = {
        "status": "complete", "years": list(YEARS), "points_per_year": 1000,
        "source_rows": 3000, "new_model_solves": 0,
        "alpha_domain": [0.25, 1.0], "gamma_domain": [0.4, 1.0],
        "projection": "isometric display projection; axes retain actual alpha and gamma values",
        "front_vertex": "front corner is alpha=0.25, gamma=0.40",
        "axes": "0.5 pt black u/v and both onset-year axes; onset-year ticks only on left axis; outward-only oblique base ticks; no plane outlines or internal grid",
        "surface_rendering": "rasterized at 600 dpi inside transparent SVG to avoid polygon seam artifacts",
        "color_scale_wanyuan": scale,
        "outside_hull_nearest_fill_cells_per_year": fill_counts,
        "baseline_npv_wanyuan": float(summary["baseline_npv"]) / 1e4,
    }
    (output / "figure_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "figure_note.md").write_text(
        "# 第1、6、11年实际NPV等轴测叠层热力图\n\n"
        "三个完整菱形采用等轴测投影；底面坐标直接标注实际范围"
        "α=0.25–1.00、γ=0.40–1.00，不使用归一化起点。"
        "蓝色代表低NPV、红色代表高NPV，三层与第6年二维热力图共用线性色标，等额差对应等距刻度。"
        "每层使用正式LHS的1000个场景结果，凸包内线性插值、边缘最近邻补全；没有新增求解。"
        "底面仅一组黑色输入轴，刻度平行于另一条底面斜轴；"
        "年份轴分别位于热力面的左右端点，轴和刻度线宽均为0.5 pt。"
        "热力面没有内网格或边框，SVG中高分辨率嵌入色面以避免多边形白缝。"
        "SVG透明无文字，保留绘图线和刻度线供PPT补标。\n",
        encoding="utf-8")


def load_sensitivity_data(source: Path) -> tuple[list[dict[str, str]], dict]:
    report = json.loads((source / "run_summary.json").read_text(encoding="utf-8"))
    if (report.get("status") != "complete"
            or report.get("capacity_rule") != "每个非基准情景重新优化光伏与储能容量"
            or not report.get("checks", {}).get("all_cases_optimal_and_audited", False)):
        raise RuntimeError("逐情景容量重优化结果尚未完成或未通过审计")
    with (source / "sensitivity_results.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != report["case_count"]:
        raise RuntimeError("情景明细数量与汇总不一致")
    return rows, report


def _level(row: dict[str, str]) -> str:
    group = row["group"]
    if group == "光伏投资成本":
        return f"−{100 * (1 - float(row['pv_capex_yuan_per_kw']) / 3800):.0f}%"
    if group == "储能投资成本":
        return f"−{100 * (1 - float(row['ess_capex_yuan_per_kwh']) / 1000):.0f}%"
    if group == "合同价格系数":
        return f"{float(row['gamma']):.2f}"
    if group == "需量分成":
        return f"{100 * float(row['demand_share']):.0f}%"
    if group == "分时电价":
        return f"{100 * (float(row['tou_multiplier']) - 1):+.0f}%"
    if group == "上网电价":
        return f"−{100 * (1 - float(row['export_price_yuan_per_kwh']) / 0.35):.1f}%"
    return f"{100 * float(row['discount_rate']):.0f}%"


def _hide_sensitivity_text(fig: mpl.figure.Figure) -> None:
    fig.patch.set_alpha(0.0)
    for ax in fig.axes:
        ax.set_facecolor("none")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title("")
        ax.tick_params(axis="both", which="both", labelbottom=False, labelleft=False,
                       labeltop=False, labelright=False)
        for item in ax.texts:
            item.set_visible(False)
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
    for item in fig.texts:
        item.set_visible(False)


def draw_sensitivity(rows: list[dict[str, str]], report: dict, *, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    with (output / "figure_source_data.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    baseline = next(row for row in rows if row["case_id"] == "baseline")
    grouped = {group: [row for row in rows if row["group"] == group] for group in GROUPS}
    if any(not members for members in grouped.values()):
        raise RuntimeError("缺少敏感性因素情景")
    ordered = sorted(
        GROUPS,
        key=lambda group: max(abs(float(row["delta_npv_yuan"])) for row in grouped[group]),
        reverse=True,
    )
    specs = [
        ("npv_yuan", 1e4, float(baseline["npv_yuan"]) / 1e4, "投资方实际 NPV（万元）"),
        ("pv_kw", 1.0, float(baseline["pv_kw"]), "最优光伏容量（kW）"),
        ("ess_kwh", 1.0, float(baseline["ess_kwh"]), "最优储能容量（kWh）"),
    ]
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial", "DejaVu Sans"],
        "svg.fonttype": "none", "pdf.fonttype": 42,
        "font.size": 9, "axes.labelsize": 8.5, "axes.titlesize": 9.5,
        "axes.linewidth": 0.8, "axes.spines.top": True, "axes.spines.right": True,
    })
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.25), sharey=True, facecolor="white")
    y_positions = np.arange(len(ordered), dtype=float)
    for panel, (ax, (key, scale, reference, title)) in enumerate(zip(axes, specs)):
        all_values = [reference]
        for y, group in zip(y_positions, ordered):
            samples = grouped[group]
            values = [reference, *[float(row[key]) / scale for row in samples]]
            lo, hi = min(values), max(values)
            all_values.extend(values)
            if hi - lo > 1e-9:
                ax.barh(y, hi - lo, left=lo, height=0.47, color=LIGHT_BLUE,
                        alpha=0.78, edgecolor="none", zorder=2)
        spread = max(all_values) - min(all_values)
        pad = max(1.0 if panel == 0 else 12.0, 0.10 * spread)
        ax.set_xlim(min(all_values) - pad, max(all_values) + pad)
        if panel == 0:
            # 278.51 万元的基准线位于 250 与 300 两个刻度之间。
            ax.set_xlim(200, 500)
            ax.set_xticks(np.arange(200, 501, 50))
        ax.axvline(reference, color=DARK, lw=1.0, ls=(0, (4, 2)), zorder=3)
        ax.set_title(title, color=DARK, pad=12)
        ax.set_ylim(-0.6, len(ordered) - 0.4)
        ax.invert_yaxis()
        ax.grid(False)
        ax.tick_params(axis="x", labelsize=7.5, length=4)
        ax.tick_params(axis="y", length=4, labelsize=8.7, pad=6)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.8)
            spine.set_color("#222222")
        ax.tick_params(axis="x", top=False)
        ax.tick_params(axis="y", right=False)
        if panel == 0:
            ax.set_yticks(y_positions, ordered)
        else:
            ax.tick_params(axis="y", left=True, labelleft=False, length=4)
    fig.suptitle("基准方案单因素敏感性：逐情景重新优化容量", x=0.05,
                 y=0.995, ha="left", fontsize=11, color=DARK)
    fig.text(0.05, 0.025,
             "浅蓝色表示测试水平的结果范围，虚线为基准值；各因素按最大绝对 NPV 变化排序。",
             fontsize=7.2, color="#485463")
    fig.subplots_adjust(left=0.16, right=0.985, top=0.83, bottom=0.15, wspace=0.14)
    stem = output / "fig5_sensitivity_reoptimized_actual_npv_ticks"
    fig.savefig(stem.with_suffix(".png"), dpi=320, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    _hide_sensitivity_text(fig)
    fig.savefig(stem.with_suffix(".svg"), transparent=True, facecolor="none",
                edgecolor="none", bbox_inches="tight")
    plt.close(fig)
    with (output / "factor_order.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["rank", "factor", "max_abs_delta_npv_yuan", "levels"])
        for rank, group in enumerate(ordered, start=1):
            writer.writerow([rank, group,
                             max(abs(float(row["delta_npv_yuan"])) for row in grouped[group]),
                             "、".join(_level(row) for row in grouped[group])])
    (output / "figure_note.md").write_text(
        "# 图5｜逐情景重优化单因素敏感性\n\n"
        "每个因素一行，三栏分别为投资方实际NPV、最优光伏容量和最优储能容量。"
        "范围条包含基准值与该因素所有测试水平的结果，基准线为虚线；具体水平见源数据表。"
        "排序依据为各因素最大绝对NPV变化；所有范围条统一淡蓝色。"
        "光伏和储能投资成本仅测试下降10%和20%；合同价格系数测试0.80、0.85、0.90、0.95；"
        "需量分成测试10%、15%、20%；分时电价测试±2%；折现率测试4%、5%、6%。"
        + ("上网电价 FiT 另测试下降2.5%和5%。" if "上网电价" in ordered else "") +
        "每个情景只改变一个输入，重新规划光伏和储能容量；零变化的基准水平没有额外求解。"
        "默认使用SOS互斥表达；需量分成15%和折现率6%使用数学上等价的binary互斥表达补算，"
        "均获认证最优解并通过同一审计。"
        "透明SVG保留图形及刻度线，隐藏所有文字，供PPT补标。\n",
        encoding="utf-8",
    )


def all_figures(root: Path = cfg.OUTPUT_ROOT, *, output: Path | None = None) -> dict:
    """已有哪组结果就绘制哪组；缺失结果会明确列出，不启动任何计算。"""
    root = Path(root)
    output = Path(output) if output is not None else root / "figures"
    drawn, missing = [], []
    boundary = root / "f2_e1" / "e1_state_boundaries.json"
    if boundary.exists():
        metadata = json.loads(boundary.read_text(encoding="utf-8"))
        if metadata.get("classification_version") == "vp-vb-value-priority-v5-gamma-tou-actual-investor-npv":
            state_boundaries_plot(boundary, output=output / "boundaries")
            drawn.append("boundaries")
        else:
            missing.append("boundaries: 结果版本与正式边界版本不匹配")
    else:
        missing.append(str(boundary))
    baseline = root / "pv_priority_baseline" / "baseline_result.json"
    if baseline.exists():
        weekly_dispatch(baseline, output=output / "dispatch")
        drawn.append("dispatch")
    else:
        missing.append(str(baseline))
    lhs = root / "locked_lhs_export_cap_1000_gamma1"
    if (lhs / "timing_results.csv").exists():
        rows, n, baseline_npv = load_onset_data(lhs)
        draw_onset_npv(rows, n, baseline_npv, output=output / "onset_npv")
        groups, summary = load_surface_data(lhs)
        draw_npv_surfaces(groups, summary, source=lhs, output=output / "npv_surfaces")
        drawn.extend(("onset_npv", "npv_surfaces"))
    else:
        missing.append(str(lhs / "timing_results.csv"))
    sensitivity = root / "sensitivity_reoptimized_with_fit"
    if (sensitivity / "sensitivity_results.csv").exists():
        rows, report = load_sensitivity_data(sensitivity)
        draw_sensitivity(rows, report, output=output / "sensitivity")
        drawn.append("sensitivity")
    else:
        missing.append(str(sensitivity / "sensitivity_results.csv"))
    return {"status": "complete" if drawn else "no_results", "drawn": drawn,
            "missing_results": missing, "output": str(output)}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=cfg.OUTPUT_ROOT)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    print(json.dumps(all_figures(args.root, output=args.out), ensure_ascii=False, indent=2))
