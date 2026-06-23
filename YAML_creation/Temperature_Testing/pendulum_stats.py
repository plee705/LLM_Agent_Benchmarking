#!/usr/bin/env python3
"""Generate summary statistics and plots from pendulum_analysis.csv."""

from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PLOTS_DIR = SCRIPT_DIR / "statistics_plots"
DEFAULT_CSV_NAME = "pendulum_analysis.csv"
MISSING_PENDULUM_XML_MSG = "pendulum.xml not found"

PRIMARY_TEXT_COLOR = "#4D4D4F"
SECONDARY_COLOR = "#BFBFBF"
PRIMARY_PLOT_COLOR_1 = "#324458"
PRIMARY_PLOT_COLOR_2 = "#F47B20"
ACCESSORY_LINE_COLOR = "#868687"

EXPECTED_BOOL_COLUMNS = [
    "xml_parse_error",
    "mujoco_compiles",
    "friction_attr_error",
    "deprecated_coordinate",
    "deprecated_angle",
    "has_worldbody",
    "has_joint",
    "has_rod_geom",
    "has_bob_body",
    "has_bob_geom",
    "gravity_correct",
    "hinge_axis_correct",
    "rod_bob_pos_match",
    "rod_mass_zero",
    "bob_mass_nonzero",
    "timed_out",
]

NAME_COLUMNS = [
    "pendulum_body_name",
    "joint_name",
    "rod_geom_name",
    "bob_body_name",
    "bob_geom_name",
]

PERCENT_COLUMNS = [
    ("xml_parse_error_pct", "XML Parse Errors", "xml_parse_error"),
    ("mujoco_compile_success_pct", "Successful MuJoCo Compilations", "mujoco_compiles"),
    ("extra_xml_files_pct", "Folders With Extra XML Files", "has_extra_xml_files"),
    ("gravity_correct_pct", "Correct Gravity Vector", "gravity_correct"),
    ("hinge_axis_correct_pct", "Correct Hinge Vectors", "hinge_axis_correct"),
    ("rod_bob_pos_match_pct", "Rod/Bob Position Matches", "rod_bob_pos_match"),
    ("rod_mass_zero_pct", "Rod Mass Zero", "rod_mass_zero"),
    ("bob_mass_nonzero_pct", "Bob Mass Nonzero", "bob_mass_nonzero"),
    ("extra_unexpected_pct", "Files With Unexpected Features", "has_extra_unexpected"),
    ("file_creation_success_pct", "Successful File Creation", "file_created"),
]


def latest_analysis_csv_path(script_dir: Path = SCRIPT_DIR) -> Path:
    candidates: list[Path] = []
    for csv_path in script_dir.glob("pendulum_analysis*.csv"):
        if csv_path.is_file():
            candidates.append(csv_path)

    if not candidates:
        return script_dir / DEFAULT_CSV_NAME

    return max(candidates, key=lambda path: (path.stat().st_mtime_ns, path.name))


def resolve_csv_path(csv_arg: str, script_dir: Path = SCRIPT_DIR) -> Path:
    candidate = Path(csv_arg).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()

    if candidate.exists():
        return candidate.resolve()

    script_relative = script_dir / candidate
    if script_relative.exists():
        return script_relative.resolve()

    return candidate.resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "csv_path",
        nargs="?",
        default=str(latest_analysis_csv_path()),
        help="Path to pendulum_analysis CSV (defaults to latest generated file)",
    )
    parser.add_argument(
        "--plots-dir",
        default=str(DEFAULT_PLOTS_DIR),
        help="Directory for generated plots",
    )
    return parser.parse_args()


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    for column in ["temperature", "xml_file_count", "elapsed_sec", "total_tokens", "input_tokens", "output_tokens"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    for column in EXPECTED_BOOL_COLUMNS:
        if column in df.columns:
            df[column] = normalize_bool_series(df[column])

    string_columns = [
        "extra_xml_files",
        "xml_parse_error_msg",
        "mujoco_compile_error",
        "gravity",
        "hinge_axis",
        "rod_geom_type",
        "joint_type",
        *NAME_COLUMNS,
    ]
    for column in string_columns:
        if column in df.columns:
            df[column] = df[column].fillna("").astype(str).str.strip()

    df["pendulum_xml_present"] = df["xml_parse_error_msg"].ne(MISSING_PENDULUM_XML_MSG)
    df["has_extra_xml_files"] = df["xml_file_count"].fillna(0).gt(1)
    df["has_extra_unexpected"] = pd.to_numeric(df["extra_unexpected"], errors="coerce").fillna(0).gt(0)
    # File creation is independent of timeout/functionality: any XML in folder counts.
    df["file_created"] = df["xml_file_count"].fillna(0).gt(0)
    df["compile_error_present"] = df["mujoco_compile_error"].ne("")
    df["compiled_file"] = df["pendulum_xml_present"] & df["mujoco_compiles"].fillna(False)
    df["completed_before_timeout"] = df["timed_out"].eq(False)
    # Converged tracks runs that completed before timeout with the target XML present.
    df["converged_file"] = df["completed_before_timeout"] & df["pendulum_xml_present"]

    return df.sort_values(["temperature", "test_num"], kind="stable").reset_index(drop=True)


def normalize_bool_series(series: pd.Series) -> pd.Series:
    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "yes": True,
        "no": False,
        "": pd.NA,
        "nan": pd.NA,
        "none": pd.NA,
        "<na>": pd.NA,
    }
    normalized = series.fillna("").astype(str).str.strip().str.lower().map(mapping)
    return normalized.astype("boolean")


def pct_true(series: pd.Series, denominator_mode: str = "all") -> float:
    if denominator_mode == "known":
        valid = series.dropna()
        if len(valid) == 0:
            return math.nan
        return 100.0 * float(valid.eq(True).mean())
    if len(series) == 0:
        return math.nan
    return 100.0 * float(series.fillna(False).eq(True).mean())


def pct_true_masked(series: pd.Series, denominator_mask: pd.Series) -> float:
    valid_rows = denominator_mask.fillna(False)
    if int(valid_rows.sum()) == 0:
        return math.nan
    return pct_true(series.loc[valid_rows])


def summarize_names(series: pd.Series) -> list[tuple[str, int, float]]:
    values = series.fillna("").astype(str).str.strip()
    values = values[values.ne("")]
    total = len(values)
    if total == 0:
        return []
    counts = Counter(values)
    return [
        (name, count, 100.0 * count / total)
        for name, count in counts.most_common()
    ]


def grouped_name_uniques(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for temperature, group in df.groupby("temperature", sort=True):
        row = {"temperature": temperature}
        for column in NAME_COLUMNS:
            values = group[column].fillna("").astype(str).str.strip()
            row[column] = int(values[values.ne("")].nunique())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("temperature")


def summarize_compile_errors(df: pd.DataFrame) -> list[tuple[str, int, float]]:
    errors = df.loc[df["mujoco_compile_error"].ne(""), "mujoco_compile_error"]
    total = len(errors)
    if total == 0:
        return []
    counts = Counter(errors)
    return [(message, count, 100.0 * count / total) for message, count in counts.most_common()]


def summary_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    def build_row(label: str, group: pd.DataFrame) -> dict[str, float | str]:
        compile_eligible = group["pendulum_xml_present"]
        timeout_rows = group["timed_out"].eq(True)
        row: dict[str, float | str] = {
            "scope": label,
            "folders": int(len(group)),
        }
        for key, _, column in PERCENT_COLUMNS:
            if column == "mujoco_compiles":
                row[key] = pct_true_masked(group[column], compile_eligible)
            else:
                row[key] = pct_true(group[column])
        row["compiled_timeout_pct"] = pct_true_masked(group["compiled_file"], timeout_rows & compile_eligible)
        row["converged_timeout_pct"] = pct_true(group.loc[timeout_rows, "converged_file"])
        row["created_timeout_pct"] = pct_true(group.loc[timeout_rows, "file_created"])
        row["compiled_file_pct"] = pct_true_masked(group["compiled_file"], compile_eligible)
        row["converged_file_pct"] = pct_true(group["converged_file"])
        row["created_file_pct"] = pct_true(group["file_created"])
        return row

    rows.append(build_row("global", df))
    for temperature, group in df.groupby("temperature", sort=True):
        rows.append(build_row(f"temperature={temperature:.1f}", group))
    return pd.DataFrame(rows)


def timeout_threshold_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    total = len(df)
    created_any = df["file_created"].fillna(False)
    created_count = int(created_any.sum())
    created_pct = 100.0 * float(created_any.mean()) if total else math.nan
    compile_eligible = df["pendulum_xml_present"].fillna(False)
    compile_eligible_count = int(compile_eligible.sum())
    compiled_any = df["compiled_file"].fillna(False)
    compiled_count = int(compiled_any.sum())
    compiled_pct = 100.0 * float(compiled_count) / compile_eligible_count if compile_eligible_count else math.nan

    for threshold in range(150, -10, -10):
        under_threshold = df["elapsed_sec"].le(threshold).fillna(False)
        converged_under_threshold = under_threshold & df["converged_file"].fillna(False)
        rows.append(
            {
                "threshold_sec": threshold,
                "created_files": created_count,
                "created_pct": created_pct,
                # We do not have per-file compile timestamps, only final compile status.
                "compiled_files": compiled_count,
                "compiled_pct": compiled_pct,
                "converged_files": int(converged_under_threshold.sum()),
                "converged_pct": 100.0 * float(converged_under_threshold.mean()) if total else math.nan,
                "compiled_among_created_pct": (
                    100.0 * float(compiled_count) / created_count if created_count else math.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def timeout_threshold_table_by_temperature(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for temperature, group in df.groupby("temperature", sort=True):
        total = len(group)
        created_any = group["file_created"].fillna(False)
        created_count = int(created_any.sum())
        created_pct = 100.0 * float(created_any.mean()) if total else math.nan
        compile_eligible = group["pendulum_xml_present"].fillna(False)
        compile_eligible_count = int(compile_eligible.sum())
        compiled_any = group["compiled_file"].fillna(False)
        compiled_count = int(compiled_any.sum())
        compiled_pct = 100.0 * float(compiled_count) / compile_eligible_count if compile_eligible_count else math.nan
        for threshold in range(150, -10, -10):
            under_threshold = group["elapsed_sec"].le(threshold).fillna(False)
            converged_under_threshold = under_threshold & group["converged_file"].fillna(False)
            rows.append(
                {
                    "temperature": temperature,
                    "threshold_sec": threshold,
                    "created_files": created_count,
                    "created_pct": created_pct,
                    # Threshold-specific converge data exists, but compile data is final-only.
                    "compiled_files": compiled_count,
                    "compiled_pct": compiled_pct,
                    "converged_files": int(converged_under_threshold.sum()),
                    "converged_pct": 100.0 * float(converged_under_threshold.mean()) if total else math.nan,
                    "compiled_among_created_pct": (
                        100.0 * float(compiled_count) / created_count if created_count else math.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def timeout_calc1_by_temperature(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for temperature, group in df.groupby("temperature", sort=True):
        timeout_group = group[group["timed_out"].eq(True)]
        rows.append(
            {
                "temperature": temperature,
                "timed_out_count": int(len(timeout_group)),
                "compiled_timeout_pct": pct_true(timeout_group["mujoco_compiles"]),
                "converged_timeout_pct": pct_true(timeout_group["converged_file"]),
                "created_timeout_pct": pct_true(timeout_group["file_created"]),
            }
        )
    return pd.DataFrame(rows).sort_values("temperature")


def lines_for_distribution(title: str, values: list[tuple[str, int, float]]) -> list[str]:
    lines = [title, "-" * 80]
    if values:
        for name, count, pct in values:
            lines.append(f"- {name}: {count} ({pct:.2f}%)")
    else:
        lines.append("- None")
    lines.append("")
    return lines


def _style_axis(ax: plt.Axes) -> None:
    for spine in ax.spines.values():
        spine.set_color(PRIMARY_TEXT_COLOR)
    ax.tick_params(colors=PRIMARY_TEXT_COLOR)
    ax.xaxis.label.set_color(PRIMARY_TEXT_COLOR)
    ax.yaxis.label.set_color(PRIMARY_TEXT_COLOR)
    ax.title.set_color(PRIMARY_TEXT_COLOR)
    ax.grid(axis="y", linestyle=":", color=SECONDARY_COLOR, alpha=0.9)


def _temperature_gray_color(temp: float) -> tuple[float, float, float]:
    temp_norm = float(max(0.0, min(1.0, temp)))
    gray = 0.88 * (1.0 - temp_norm)
    return (gray, gray, gray)


def _temperature_marker(temp: float) -> str:
    marker_map = {
        0.0: "o",
        0.1: "s",
        0.2: "^",
        0.3: "v",
        0.4: "D",
        0.5: "*",
        0.7: "p",
        1.0: "H",
    }
    return marker_map.get(float(temp), "o")


def _spread_positions(values: list[float], min_gap: float = 0.08, lower: float = -1.02, upper: float = 1.02) -> list[float]:
    if not values:
        return []
    positions = sorted(values)
    for idx in range(1, len(positions)):
        if positions[idx] - positions[idx - 1] < min_gap:
            positions[idx] = positions[idx - 1] + min_gap
    if positions[-1] > upper:
        shift = positions[-1] - upper
        positions = [value - shift for value in positions]
    if positions[0] < lower:
        shift = lower - positions[0]
        positions = [value + shift for value in positions]
    return positions


def _annotate_pie_with_leaders(ax: plt.Axes, wedges, labels: list[str], values: list[int]) -> None:
    total = sum(values)
    if total == 0:
        return

    points = []
    for wedge, _label, value in zip(wedges, labels, values):
        angle = (wedge.theta2 + wedge.theta1) / 2.0
        radians = math.radians(angle)
        x = math.cos(radians)
        y = math.sin(radians)
        pct = 100.0 * value / total
        points.append(
            {
                "x": x,
                "y": y,
                "label": f"{pct:.1f}%",
            }
        )

    left = [p for p in points if p["x"] < 0]
    right = [p for p in points if p["x"] >= 0]

    # Keep labels just outside the pie radius so text never enters wedges.
    for group, x_text in ((left, -1.06), (right, 1.06)):
        group.sort(key=lambda p: p["y"])
        original_ys = [p["y"] for p in group]
        adjusted_ys = _spread_positions(original_ys)
        for point, y_text in zip(group, adjusted_ys):
            ha = "right" if x_text < 0 else "left"
            ax.annotate(
                point["label"],
                xy=(1.0 * point["x"], 1.0 * point["y"]),
                xytext=(x_text, y_text),
                ha=ha,
                va="center",
                color=PRIMARY_TEXT_COLOR,
                fontsize=12,
                arrowprops={
                    "arrowstyle": "-",
                    "color": ACCESSORY_LINE_COLOR,
                    "lw": 1.0,
                    "shrinkA": 0,
                    "shrinkB": 0,
                    "connectionstyle": "arc3,rad=0.0",
                },
            )


def pie_error_breakdown(df: pd.DataFrame, plots_dir: Path) -> None:
    missing = int(df["file_created"].eq(False).sum())
    xml_errors = int(df["xml_parse_error"].fillna(False).sum())
    compiled = int(df["mujoco_compiles"].fillna(False).sum())
    compile_failures = int(
        (
            df["file_created"].eq(True)
            & df["xml_parse_error"].fillna(False).eq(False)
            & df["mujoco_compiles"].fillna(False).eq(False)
        ).sum()
    )
    labels = ["Compiled", "XML Parse Error", "MuJoCo Compile Error", "Missing XML"]
    values = [compiled, xml_errors, compile_failures, missing]
    filtered = [(label, value) for label, value in zip(labels, values) if value > 0]
    values = [value for _, value in filtered]
    labels = [label for label, _ in filtered]
    fig, ax = plt.subplots(figsize=(10, 6.3))
    colors = [PRIMARY_PLOT_COLOR_1, PRIMARY_PLOT_COLOR_2, ACCESSORY_LINE_COLOR, SECONDARY_COLOR][: len(values)]
    wedges, _ = ax.pie(
        values,
        labels=None,
        startangle=90,
        colors=colors,
        wedgeprops={"edgecolor": "white", "linewidth": 1.2},
    )

    # Legend keeps category-color mapping explicit while labels show percentages.
    legend = ax.legend(
        wedges,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.04),
        ncol=2,
        frameon=False,
    )
    if legend is not None:
        legend.get_title().set_color(PRIMARY_TEXT_COLOR)
        for text in legend.get_texts():
            text.set_color(PRIMARY_TEXT_COLOR)

    _annotate_pie_with_leaders(ax, wedges, labels, values)
    ax.set_title("Pendulum XML Outcome Breakdown", fontsize=16, pad=20)
    ax.set_aspect("equal")
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-1.15, 1.15)
    fig.tight_layout()
    fig.savefig(plots_dir / "pie_outcomes.png", dpi=200, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def bar_plot(df: pd.DataFrame, x: str, y: str, ylabel: str, title: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(df[x].astype(str), df[y], color=PRIMARY_PLOT_COLOR_1)
    ax.set_xlabel("Temperature")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(bottom=0)
    _style_axis(ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def grouped_bar_plot(df: pd.DataFrame, output_path: Path) -> None:
    categories = [column for column in NAME_COLUMNS]
    category_labels = {
        "pendulum_body_name": "Pendulum Body",
        "joint_name": "Joint",
        "rod_geom_name": "Rod Geom",
        "bob_body_name": "Bob Body",
        "bob_geom_name": "Bob Geom",
    }
    colors = ["#1F4E79", "#4F81BD", "#F47B20", "#F9A65A", "#6E6E6E"]
    x_positions = list(range(len(df)))
    width = 0.15
    fig, ax = plt.subplots(figsize=(11, 6))
    for idx, column in enumerate(categories):
        offsets = [x + (idx - 2) * width for x in x_positions]
        ax.bar(offsets, df[column], width=width, label=category_labels[column], color=colors[idx])
    ax.set_xticks(x_positions)
    ax.set_xticklabels([f"{temp:.1f}" for temp in df["temperature"]])
    ax.set_xlabel("Temperature")
    ax.set_ylabel("Unique name count")
    ax.set_title("Different Variable Names by Temperature")
    ax.legend(frameon=False)
    _style_axis(ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def single_metric_line_plot(
    df: pd.DataFrame,
    metric_column: str,
    title: str,
    y_label: str,
    output_path: Path,
    color: str,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df["temperature"], df[metric_column], marker="o", color=color)
    ax.set_xlabel("Temperature")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.set_ylim(0, 100)
    _style_axis(ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_timeout_calc1_global(metrics: pd.DataFrame, output_path: Path) -> None:
    global_row = metrics.loc[metrics["scope"].eq("global")].iloc[0]
    labels = ["Compiled", "Converged"]
    values = [
        float(global_row["compiled_timeout_pct"]),
        float(global_row["converged_timeout_pct"]),
    ]
    colors = [PRIMARY_PLOT_COLOR_1, PRIMARY_PLOT_COLOR_2]

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    bars = ax.bar(labels, values, color=colors, width=0.55)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Percent")
    ax.set_title("Global XML File Statistics")
    _style_axis(ax)

    for bar, value in zip(bars, values):
        if pd.notna(value):
            y_text = min(value + 1.5, 99.0)
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                y_text,
                f"{value:.1f}%",
                ha="center",
                va="bottom",
                color=PRIMARY_TEXT_COLOR,
                fontsize=11,
                fontweight="bold",
            )
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_timeout_calc2_global(timeout_table: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        timeout_table["threshold_sec"],
        timeout_table["compiled_pct"],
        marker="o",
        label="Compiled",
        color=PRIMARY_PLOT_COLOR_1,
    )
    ax.plot(
        timeout_table["threshold_sec"],
        timeout_table["converged_pct"],
        marker="s",
        label="Converged",
        color=PRIMARY_PLOT_COLOR_2,
    )
    ax.set_xlabel("Threshold (sec)")
    ax.set_ylabel("Percent")
    ax.set_ylim(0, 100)
    ax.set_title("Compiled and Converged by Threshold (Global)")
    ax.invert_xaxis()
    ax.legend(frameon=False)
    _style_axis(ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_timeout_calc2_metric_by_temperature(
    timeout_by_temp: pd.DataFrame,
    timeout_global: pd.DataFrame,
    metric_column: str,
    title: str,
    output_path: Path,
) -> None:
    temperatures = sorted(timeout_by_temp["temperature"].unique())
    fig, ax = plt.subplots(figsize=(11, 7))

    for temperature in temperatures:
        subset = timeout_by_temp.loc[timeout_by_temp["temperature"].eq(temperature)].sort_values("threshold_sec")
        color = _temperature_gray_color(float(temperature))
        marker = _temperature_marker(float(temperature))
        ax.plot(
            subset["threshold_sec"],
            subset[metric_column],
            marker=marker,
            linewidth=1.5,
            label=f"T={temperature:.1f}",
            color=color,
        )

    ax.plot(
        timeout_global["threshold_sec"],
        timeout_global[metric_column],
        color=PRIMARY_PLOT_COLOR_2,
        linewidth=2.5,
        linestyle="--",
        label="Global Average",
    )

    ax.set_title(title)
    ax.set_ylabel("Percent")
    ax.set_xlabel("Threshold (sec)")
    ax.set_ylim(0, 100)
    ax.invert_xaxis()
    _style_axis(ax)
    ax.legend(frameon=False, ncol=4, fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def clear_existing_outputs(output_dir: Path) -> None:
    """Overwrite behavior for same CSV: remove prior plots in this CSV folder only."""
    if not output_dir.exists():
        return

    for image_path in output_dir.glob("*"):
        if image_path.is_file() and image_path.suffix.lower() in {".png", ".jpg", ".jpeg", ".svg", ".pdf"}:
            image_path.unlink(missing_ok=True)


def build_output_dir(plots_root: Path, csv_path: Path) -> Path:
    """Create per-CSV output folder; preserve other CSV folders."""
    output_dir = plots_root / f"{csv_path.stem}_Plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    clear_existing_outputs(output_dir)
    return output_dir


def write_summary(
    df: pd.DataFrame,
    metrics: pd.DataFrame,
    timeout_table: pd.DataFrame,
    summary_path: Path,
) -> None:
    lines: list[str] = []
    lines.append("Pendulum Statistics Summary")
    lines.append("=" * 80)
    lines.append(f"Source CSV: {summary_path.parent / df.attrs['source_name']}")
    lines.append(f"Total folders analyzed: {len(df)}")
    lines.append("")

    lines.append("Global and Temperature Metrics")
    lines.append("-" * 80)
    display_metrics = metrics.copy()
    for column in display_metrics.columns:
        if column.endswith("_pct"):
            display_metrics[column] = display_metrics[column].map(format_pct)
    lines.append(display_metrics.to_string(index=False))
    lines.append("")

    lines.append("MuJoCo Compile Errors")
    lines.append("-" * 80)
    compile_errors = summarize_compile_errors(df)
    if compile_errors:
        for message, count, pct in compile_errors:
            lines.append(f"- {count} ({pct:.2f}%): {message}")
    else:
        lines.append("- None")
    lines.append("")

    lines.extend(lines_for_distribution("Pendulum Body Names", summarize_names(df["pendulum_body_name"])))
    lines.extend(lines_for_distribution("Joint Names", summarize_names(df["joint_name"])))
    lines.extend(lines_for_distribution("Rod Geom Types", summarize_names(df["rod_geom_type"])))

    lines.append("Per-Temperature Distributions")
    lines.append("-" * 80)
    for temperature, group in df.groupby("temperature", sort=True):
        lines.append(f"Temperature {temperature:.1f}")
        lines.append(f"Folders: {len(group)}")
        lines.extend(lines_for_distribution("Compile Errors", summarize_compile_errors(group)))
        lines.extend(lines_for_distribution("Pendulum Body Names", summarize_names(group["pendulum_body_name"])))
        lines.extend(lines_for_distribution("Joint Names", summarize_names(group["joint_name"])))
        lines.extend(lines_for_distribution("Rod Geom Types", summarize_names(group["rod_geom_type"])))

    lines.append("Timeout Statistics")
    lines.append("-" * 80)
    timeout_rows = df[df["timed_out"].eq(True)]
    lines.append(f"Timed out folders: {len(timeout_rows)}")
    lines.append(f"Timed out folders that compile: {format_pct(pct_true(timeout_rows['mujoco_compiles']))}")
    lines.append(f"Timed out folders that converged: {format_pct(pct_true(timeout_rows['converged_file']))}")
    lines.append(f"Timed out folders that created XML: {format_pct(pct_true(timeout_rows['file_created']))}")
    lines.append("")
    display_timeout = timeout_table.copy()
    for column in ["created_pct", "compiled_pct", "converged_pct", "compiled_among_created_pct"]:
        display_timeout[column] = display_timeout[column].map(format_pct)
    lines.append(display_timeout.to_string(index=False))
    lines.append("")

    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_pct(value: float) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{value:.2f}%"


def generate_plots(
    df: pd.DataFrame,
    metrics: pd.DataFrame,
    timeout_table: pd.DataFrame,
    plot_dir: Path,
) -> Path:
    pie_error_breakdown(df, plot_dir)

    by_temp = metrics.loc[metrics["scope"].ne("global")].copy()
    by_temp["temperature"] = by_temp["scope"].str.replace("temperature=", "", regex=False).astype(float)
    by_temp = by_temp.sort_values("temperature")

    bar_plot(
        by_temp,
        "temperature",
        "xml_parse_error_pct",
        "Percent",
        "XML Parse Error Rate by Temperature",
        plot_dir / "bar_xml_parse_error_by_temperature.png",
    )
    bar_plot(
        by_temp,
        "temperature",
        "mujoco_compile_success_pct",
        "Percent",
        "MuJoCo Compilation Success by Temperature",
        plot_dir / "bar_mujoco_compile_by_temperature.png",
    )
    bar_plot(
        by_temp,
        "temperature",
        "extra_xml_files_pct",
        "Percent",
        "Extra XML Files by Temperature",
        plot_dir / "bar_extra_xml_by_temperature.png",
    )
    bar_plot(
        by_temp,
        "temperature",
        "extra_unexpected_pct",
        "Percent",
        "Unexpected Features by Temperature",
        plot_dir / "bar_unexpected_features_by_temperature.png",
    )

    unique_names = grouped_name_uniques(df)
    grouped_bar_plot(unique_names, plot_dir / "bar_unique_variable_names_by_temperature.png")
    single_metric_line_plot(
        by_temp,
        metric_column="compiled_file_pct",
        title="Compiled File (%) by Temperature",
        y_label="Percent of folders",
        output_path=plot_dir / "line_compiled_by_temperature.png",
        color=PRIMARY_PLOT_COLOR_1,
    )
    single_metric_line_plot(
        by_temp,
        metric_column="converged_file_pct",
        title="Converged File (%) by Temperature",
        y_label="Percent of folders",
        output_path=plot_dir / "line_converged_by_temperature.png",
        color=PRIMARY_PLOT_COLOR_2,
    )

    timeout_calc2_temp = timeout_threshold_table_by_temperature(df)
    plot_timeout_calc1_global(metrics, plot_dir / "line_timeout_calc1_global.png")
    plot_timeout_calc2_global(timeout_table, plot_dir / "line_timeout_calc2_global_threshold.png")
    plot_timeout_calc2_metric_by_temperature(
        timeout_calc2_temp,
        timeout_table,
        metric_column="compiled_pct",
        title="Compiled by Threshold (All Temperatures)",
        output_path=plot_dir / "line_timeout_calc2_compiled_by_temperature_threshold.png",
    )
    plot_timeout_calc2_metric_by_temperature(
        timeout_calc2_temp,
        timeout_table,
        metric_column="converged_pct",
        title="Converged by Threshold (All Temperatures)",
        output_path=plot_dir / "line_timeout_calc2_converged_by_temperature_threshold.png",
    )

    return plot_dir


def main() -> None:
    args = parse_args()
    csv_path = resolve_csv_path(args.csv_path)
    plots_root = Path(args.plots_dir).expanduser().resolve()
    plot_dir = build_output_dir(plots_root, csv_path)
    summary_path = plot_dir / f"{csv_path.stem}_statistics.txt"

    df = load_data(csv_path)
    df.attrs["source_name"] = csv_path.name
    metrics = summary_metrics(df)
    timeout_table = timeout_threshold_table(df)

    write_summary(df, metrics, timeout_table, summary_path)
    plot_dir = generate_plots(df, metrics, timeout_table, plot_dir)

    print(f"Summary written to: {summary_path}")
    print(f"Plots written to: {plot_dir}")


if __name__ == "__main__":
    main()