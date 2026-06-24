#!/usr/bin/env python3
"""
cutoff_compile_convergence_analysis.py

Reconstructs pendulum.xml revisions from OpenCode JSONL event logs and evaluates
how compile and convergence percentages change as the hypothetical timeout cutoff
is reduced.

Expected directory layout, matching the existing temperature test scripts:

    Temperature_Testing/
        temp_testing.csv
        V2_Temp_0.1/
            test1/ or test01/
                opencode_events.jsonl
                pendulum.xml
        V3_Temp_0.2/
            test1/
                opencode_events.jsonl
        ...

For each cutoff, the compile result uses the latest pendulum.xml write event that
occurred before the cutoff. By default, the known prompt-error correction is applied
before compiling each XML snapshot: joint friction=... is changed to frictionloss=....
This answers:

    "If OpenCode had been stopped at this cutoff, would the file on disk compile?"

Convergence uses temp_testing.csv elapsed_sec/timed_out:

    converged_by_cutoff = (not timed_out) and elapsed_sec <= cutoff

Outputs:
    cutoff_revision_history.csv
    cutoff_threshold_results.csv
    cutoff_threshold_results_by_temperature.csv
    cutoff_analysis_summary.txt
    cutoff_analysis_plots/*.png

Run:
    python3 cutoff_compile_convergence_analysis.py
    python3 cutoff_compile_convergence_analysis.py --root /path/to/Temperature_Testing
    python3 cutoff_compile_convergence_analysis.py --analysis-csv pendulum_analysis_fixedfriction.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

try:
    import mujoco as _mujoco
    MUJOCO_AVAILABLE = True
except ImportError:
    _mujoco = None
    MUJOCO_AVAILABLE = False

TEMP_PATTERN = re.compile(r"^V(\d+)_Temp_([\d.]+)$")
DEFAULT_CUTOFF_START = 150
DEFAULT_CUTOFF_STOP = 0
DEFAULT_CUTOFF_STEP = 10

PRIMARY_TEXT_COLOR = "#4D4D4F"
SECONDARY_COLOR = "#BFBFBF"
PRIMARY_PLOT_COLOR_1 = "#324458"
PRIMARY_PLOT_COLOR_2 = "#F47B20"
ACCESSORY_LINE_COLOR = "#868687"


@dataclass(frozen=True)
class TestRun:
    temp_folder: str
    temperature: float
    test_num: str
    test_dir: Path
    run_id: str
    elapsed_sec: float | None
    timed_out: bool | None


@dataclass(frozen=True)
class XmlRevision:
    temp_folder: str
    temperature: float
    test_num: str
    run_id: str
    revision_index: int
    elapsed_sec: float
    timestamp_raw: Any
    compiles: bool
    compile_error: str
    xml_chars: int
    source_file: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Root Temperature_Testing directory. Defaults to this script's folder.",
    )
    parser.add_argument(
        "--timing-csv",
        type=Path,
        default=None,
        help="CSV containing run_id, elapsed_sec, and timed_out. Defaults to ROOT/temp_testing.csv.",
    )
    parser.add_argument(
        "--analysis-csv",
        type=Path,
        default=None,
        help=(
            "Optional pendulum_analysis*.csv. Used as a fallback source for "
            "temperature/test metadata and elapsed_sec/timed_out if temp_testing.csv is absent."
        ),
    )
    parser.add_argument(
        "--events-name",
        default="opencode_events.jsonl",
        help="Name of the OpenCode JSONL event file in each test folder.",
    )
    parser.add_argument("--cutoff-start", type=int, default=DEFAULT_CUTOFF_START)
    parser.add_argument("--cutoff-stop", type=int, default=DEFAULT_CUTOFF_STOP)
    parser.add_argument("--cutoff-step", type=int, default=DEFAULT_CUTOFF_STEP)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to ROOT/cutoff_analysis_outputs.",
    )
    parser.add_argument(
        "--include-ever-compiled",
        action="store_true",
        help="Also calculate whether any revision before cutoff ever compiled.",
    )
    parser.add_argument(
        "--no-friction-fix",
        action="store_true",
        help=(
            "Disable the prompt-error correction that changes joint friction=... "
            "to frictionloss=... before MuJoCo compilation. By default this fix is applied."
        ),
    )
    return parser.parse_args()


def normalize_bool(value: Any) -> bool | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    if text in {"", "nan", "none", "<na>"}:
        return None
    return None


def to_float(value: Any) -> float | None:
    try:
        if value is None or str(value).strip() == "":
            return None
        x = float(value)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def canonical_test_num(value: Any) -> str:
    text = str(value).strip()
    if not text:
        return text
    match = re.search(r"test0*(\d+)$", text)
    if match:
        return f"test{int(match.group(1))}"
    return text


def possible_test_keys(temp_folder: str, test_num: str) -> set[tuple[str, str]]:
    keys = {(temp_folder, test_num), (temp_folder, canonical_test_num(test_num))}
    match = re.search(r"test0*(\d+)$", test_num)
    if match:
        n = int(match.group(1))
        keys.add((temp_folder, f"test{n}"))
        keys.add((temp_folder, f"test{n:02d}"))
    return keys


def resolve_path(path: Path, root: Path) -> Path:
    if path.is_absolute():
        return path
    candidate = root / path
    if candidate.exists():
        return candidate
    return path


def load_timing_lookup(csv_path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    if not csv_path.exists():
        return lookup

    with csv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            run_id = str(row.get("run_id", "")).strip()
            temp_folder = ""
            test_num = ""

            if "/" in run_id:
                temp_folder, test_num = run_id.split("/", 1)
            else:
                temp_folder = str(row.get("temp_folder", "")).strip()
                test_num = str(row.get("test_num", "")).strip()

            if not temp_folder or not test_num:
                continue

            temp_value = row.get("temp", row.get("temperature"))
            record = {
                "run_id": run_id or f"{temp_folder}/{test_num}",
                "elapsed_sec": to_float(row.get("elapsed_sec")),
                "timed_out": normalize_bool(row.get("timed_out")),
                "temperature": to_float(temp_value),
            }
            for key in possible_test_keys(temp_folder, test_num):
                lookup[key] = record

    return lookup


def load_analysis_lookup(csv_path: Path | None) -> dict[tuple[str, str], dict[str, Any]]:
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    if csv_path is None or not csv_path.exists():
        return lookup

    with csv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            temp_folder = str(row.get("temp_folder", "")).strip()
            test_num = str(row.get("test_num", "")).strip()
            if not temp_folder or not test_num:
                continue
            record = {
                "run_id": f"{temp_folder}/{test_num}",
                "elapsed_sec": to_float(row.get("elapsed_sec")),
                "timed_out": normalize_bool(row.get("timed_out")),
                "temperature": to_float(row.get("temperature")),
            }
            for key in possible_test_keys(temp_folder, test_num):
                lookup[key] = record
    return lookup


def discover_test_runs(root: Path, timing_lookup: dict, analysis_lookup: dict) -> list[TestRun]:
    runs: list[TestRun] = []

    for temp_dir in sorted(root.glob("V*_Temp_*")):
        if not temp_dir.is_dir():
            continue
        match = TEMP_PATTERN.match(temp_dir.name)
        if not match:
            continue
        folder_temp = float(match.group(2))

        for test_dir in sorted(temp_dir.glob("test*")):
            if not test_dir.is_dir():
                continue
            temp_folder = temp_dir.name
            test_num = test_dir.name
            metadata = None
            for key in possible_test_keys(temp_folder, test_num):
                metadata = timing_lookup.get(key) or analysis_lookup.get(key)
                if metadata:
                    break

            elapsed_sec = metadata.get("elapsed_sec") if metadata else None
            timed_out = metadata.get("timed_out") if metadata else None
            temperature = metadata.get("temperature") if metadata and metadata.get("temperature") is not None else folder_temp
            run_id = metadata.get("run_id") if metadata and metadata.get("run_id") else f"{temp_folder}/{test_num}"

            runs.append(
                TestRun(
                    temp_folder=temp_folder,
                    temperature=float(temperature),
                    test_num=test_num,
                    test_dir=test_dir,
                    run_id=run_id,
                    elapsed_sec=elapsed_sec,
                    timed_out=timed_out,
                )
            )

    return runs


def timestamp_to_seconds(timestamp: Any) -> float | None:
    if timestamp is None:
        return None
    if isinstance(timestamp, (int, float)):
        value = float(timestamp)
        # OpenCode JSON uses milliseconds in the observed files.
        return value / 1000.0 if value > 1.0e11 else value

    text = str(timestamp).strip()
    if not text:
        return None
    try:
        value = float(text)
        return value / 1000.0 if value > 1.0e11 else value
    except ValueError:
        pass

    try:
        iso = text.replace("Z", "+00:00")
        return datetime.fromisoformat(iso).timestamp()
    except ValueError:
        return None


def is_pendulum_xml_path(path_value: Any) -> bool:
    if path_value is None:
        return False
    return Path(str(path_value)).name == "pendulum.xml"


def extract_write_xml(event: dict[str, Any]) -> str | None:
    part = event.get("part") or {}
    if part.get("type") != "tool" or part.get("tool") != "write":
        return None

    state = part.get("state") or {}
    input_data = state.get("input") or {}
    file_path = input_data.get("filePath") or input_data.get("filepath")
    if not is_pendulum_xml_path(file_path):
        metadata = state.get("metadata") or {}
        if not is_pendulum_xml_path(metadata.get("filepath")):
            return None

    content = input_data.get("content")
    return content if isinstance(content, str) else None


def apply_prompt_error_friction_fix(xml_text: str) -> tuple[str, int]:
    """
    Correct the known prompt error before compilation.

    MuJoCo joints use frictionloss=..., not friction=.... Some test outputs used
    friction="0" because the prompt contained the wrong attribute. This function
    changes joint friction=VALUE to frictionloss=VALUE and removes friction.

    Returns the corrected XML text and the number of joint attributes changed.
    If the XML is not parseable, the original text is returned so the caller can
    report the real parse error.
    """
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return xml_text, 0

    fixes = 0
    for joint in root.iter("joint"):
        if "friction" in joint.attrib:
            friction_value = joint.attrib.pop("friction")
            if "frictionloss" not in joint.attrib:
                joint.set("frictionloss", friction_value)
            fixes += 1

    if fixes == 0:
        return xml_text, 0

    return ET.tostring(root, encoding="unicode"), fixes


def mujoco_can_load(xml_text: str, apply_friction_fix: bool = True) -> tuple[bool, str]:
    if apply_friction_fix:
        xml_text, fix_count = apply_prompt_error_friction_fix(xml_text)
    else:
        fix_count = 0

    # XML parse precheck gives clear errors even when MuJoCo is unavailable.
    try:
        ET.fromstring(xml_text)
    except ET.ParseError as exc:
        return False, f"XML parse error: {' '.join(str(exc).split())}"

    if not MUJOCO_AVAILABLE:
        return False, "MuJoCo not installed; install with: pip install mujoco"

    with tempfile.TemporaryDirectory() as tmp:
        xml_path = Path(tmp) / "pendulum.xml"
        xml_path.write_text(xml_text, encoding="utf-8")
        try:
            _mujoco.MjModel.from_xml_path(str(xml_path))
            return True, f"friction_fix_applied={fix_count}" if fix_count else ""
        except Exception as exc:
            msg = " ".join(str(exc).split())
            if fix_count:
                msg = f"friction_fix_applied={fix_count}; {msg}"
            return False, msg


def parse_revisions_for_run(run: TestRun, events_name: str, apply_friction_fix: bool = True) -> list[XmlRevision]:
    events_path = run.test_dir / events_name
    if not events_path.exists():
        return []

    raw_records: list[tuple[float, Any, str]] = []
    with events_path.open("r", encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            xml_text = extract_write_xml(event)
            if xml_text is None:
                continue
            ts_sec = timestamp_to_seconds(event.get("timestamp"))
            if ts_sec is None:
                # Fall back to the tool's recorded start time if present.
                state = ((event.get("part") or {}).get("state") or {})
                ts_sec = timestamp_to_seconds(((state.get("time") or {}).get("start")))
            if ts_sec is None:
                # Stable ordering fallback; elapsed will use ordinal seconds.
                ts_sec = float(line_number)
            raw_records.append((ts_sec, event.get("timestamp"), xml_text))

    if not raw_records:
        return []

    raw_records.sort(key=lambda item: item[0])
    start_ts = raw_records[0][0]

    revisions: list[XmlRevision] = []
    for idx, (ts_sec, timestamp_raw, xml_text) in enumerate(raw_records, start=1):
        compiles, compile_error = mujoco_can_load(xml_text, apply_friction_fix=apply_friction_fix)
        revisions.append(
            XmlRevision(
                temp_folder=run.temp_folder,
                temperature=run.temperature,
                test_num=run.test_num,
                run_id=run.run_id,
                revision_index=idx,
                elapsed_sec=max(0.0, ts_sec - start_ts),
                timestamp_raw=timestamp_raw,
                compiles=compiles,
                compile_error=compile_error,
                xml_chars=len(xml_text),
                source_file=str(events_path),
            )
        )
    return revisions


def cutoffs(start: int, stop: int, step: int) -> list[int]:
    if step <= 0:
        raise ValueError("cutoff-step must be positive")
    if start >= stop:
        return list(range(start, stop - 1, -step))
    return list(range(start, stop + 1, step))


def latest_revision_before(revisions: list[XmlRevision], cutoff: float) -> XmlRevision | None:
    eligible = [rev for rev in revisions if rev.elapsed_sec <= cutoff]
    return eligible[-1] if eligible else None


def any_compiled_before(revisions: list[XmlRevision], cutoff: float) -> bool:
    return any(rev.elapsed_sec <= cutoff and rev.compiles for rev in revisions)


def build_cutoff_rows(
    runs: list[TestRun],
    revisions_by_run: dict[str, list[XmlRevision]],
    cutoff_values: list[int],
    include_ever_compiled: bool,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []

    for run in runs:
        revisions = revisions_by_run.get(run.run_id, [])
        for cutoff in cutoff_values:
            latest = latest_revision_before(revisions, cutoff)
            compiled_at_cutoff = bool(latest.compiles) if latest is not None else False
            converged_by_cutoff = (
                run.timed_out is False
                and run.elapsed_sec is not None
                and run.elapsed_sec <= cutoff
            )
            record = {
                "temp_folder": run.temp_folder,
                "temperature": run.temperature,
                "test_num": run.test_num,
                "run_id": run.run_id,
                "cutoff_sec": cutoff,
                "elapsed_sec": run.elapsed_sec,
                "timed_out": run.timed_out,
                "converged_by_cutoff": converged_by_cutoff,
                "compiled_at_cutoff": compiled_at_cutoff,
                "revision_used": latest.revision_index if latest else None,
                "revision_elapsed_sec": latest.elapsed_sec if latest else None,
                "compile_error_at_cutoff": latest.compile_error if latest else "No pendulum.xml revision before cutoff",
            }
            if include_ever_compiled:
                record["ever_compiled_before_cutoff"] = any_compiled_before(revisions, cutoff)
            records.append(record)

    return pd.DataFrame.from_records(records)


def summarize_thresholds(cutoff_rows: pd.DataFrame, include_ever_compiled: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    agg_spec = {
        "tests": ("run_id", "count"),
        "compiled_count": ("compiled_at_cutoff", "sum"),
        "converged_count": ("converged_by_cutoff", "sum"),
    }
    if include_ever_compiled:
        agg_spec["ever_compiled_count"] = ("ever_compiled_before_cutoff", "sum")

    global_df = cutoff_rows.groupby("cutoff_sec", as_index=False).agg(**agg_spec)
    global_df["compiled_pct"] = 100.0 * global_df["compiled_count"] / global_df["tests"]
    global_df["converged_pct"] = 100.0 * global_df["converged_count"] / global_df["tests"]
    if include_ever_compiled:
        global_df["ever_compiled_pct"] = 100.0 * global_df["ever_compiled_count"] / global_df["tests"]
    global_df = global_df.sort_values("cutoff_sec", ascending=False)

    by_temp = cutoff_rows.groupby(["temperature", "cutoff_sec"], as_index=False).agg(**agg_spec)
    by_temp["compiled_pct"] = 100.0 * by_temp["compiled_count"] / by_temp["tests"]
    by_temp["converged_pct"] = 100.0 * by_temp["converged_count"] / by_temp["tests"]
    if include_ever_compiled:
        by_temp["ever_compiled_pct"] = 100.0 * by_temp["ever_compiled_count"] / by_temp["tests"]
    by_temp = by_temp.sort_values(["temperature", "cutoff_sec"], ascending=[True, False])

    return global_df, by_temp


def revision_dataframe(revisions: list[XmlRevision]) -> pd.DataFrame:
    return pd.DataFrame.from_records([rev.__dict__ for rev in revisions])


def _style_axis(ax: plt.Axes) -> None:
    for spine in ax.spines.values():
        spine.set_color(PRIMARY_TEXT_COLOR)
    ax.tick_params(colors=PRIMARY_TEXT_COLOR)
    ax.xaxis.label.set_color(PRIMARY_TEXT_COLOR)
    ax.yaxis.label.set_color(PRIMARY_TEXT_COLOR)
    ax.title.set_color(PRIMARY_TEXT_COLOR)
    ax.grid(axis="y", linestyle=":", color=SECONDARY_COLOR, alpha=0.9)


def plot_global_thresholds(global_df: pd.DataFrame, output_path: Path, include_ever_compiled: bool) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(global_df["cutoff_sec"], global_df["compiled_pct"], marker="o", label="Compiles at cutoff", color=PRIMARY_PLOT_COLOR_1)
    ax.plot(global_df["cutoff_sec"], global_df["converged_pct"], marker="s", label="Converged by cutoff", color=PRIMARY_PLOT_COLOR_2)
    if include_ever_compiled and "ever_compiled_pct" in global_df.columns:
        ax.plot(global_df["cutoff_sec"], global_df["ever_compiled_pct"], marker="^", linestyle="--", label="Ever compiled before cutoff", color=ACCESSORY_LINE_COLOR)
    ax.set_xlabel("Cutoff time (sec)")
    ax.set_ylabel("Percent of tests")
    ax.set_title("Compile and Convergence Rate vs Cutoff Time")
    ax.set_ylim(0, 100)
    ax.invert_xaxis()
    ax.legend(frameon=False)
    _style_axis(ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _temperature_gray_color(temp: float) -> tuple[float, float, float]:
    temp_norm = float(max(0.0, min(1.0, temp)))
    gray = 0.88 * (1.0 - temp_norm)
    return (gray, gray, gray)


def _temperature_orange_color(temp: float) -> tuple[float, float, float]:
    """Burnt orange (low temp) → light orange (high temp)."""
    temp_norm = float(max(0.0, min(1.0, temp)))
    dark = (0.50, 0.18, 0.02)   # ~#801C05
    light = (1.00, 0.78, 0.50)  # ~#FFC87F
    return (
        dark[0] + temp_norm * (light[0] - dark[0]),
        dark[1] + temp_norm * (light[1] - dark[1]),
        dark[2] + temp_norm * (light[2] - dark[2]),
    )


def _temperature_marker(temp: float) -> str:
    marker_map = {0.0: "o", 0.1: "s", 0.2: "^", 0.3: "v", 0.4: "D", 0.5: "*", 0.7: "p", 1.0: "H"}
    return marker_map.get(float(temp), "o")


def plot_metric_by_temperature(
    by_temp: pd.DataFrame,
    metric_col: str,
    title: str,
    output_path: Path,
    color_fn=None,
) -> None:
    if color_fn is None:
        color_fn = _temperature_gray_color
    fig, ax = plt.subplots(figsize=(11, 7))
    for temperature in sorted(by_temp["temperature"].dropna().unique()):
        subset = by_temp.loc[by_temp["temperature"].eq(temperature)].sort_values("cutoff_sec", ascending=False)
        ax.plot(
            subset["cutoff_sec"],
            subset[metric_col],
            marker=_temperature_marker(float(temperature)),
            color=color_fn(float(temperature)),
            linewidth=1.6,
            label=f"T={temperature:g}",
        )
    ax.set_xlabel("Cutoff time (sec)")
    ax.set_ylabel("Percent of tests")
    ax.set_title(title)
    ax.set_ylim(0, 100)
    ax.invert_xaxis()
    ax.legend(frameon=False, ncol=4, fontsize=8)
    _style_axis(ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_combined_by_temperature(by_temp: pd.DataFrame, output_path: Path) -> None:
    """Overlay convergence (grayscale) and compile (orange scale) lines for every temperature."""
    fig, ax = plt.subplots(figsize=(12, 7))
    temperatures = sorted(by_temp["temperature"].dropna().unique())
    for temperature in temperatures:
        subset = by_temp.loc[by_temp["temperature"].eq(temperature)].sort_values("cutoff_sec", ascending=False)
        marker = _temperature_marker(float(temperature))
        ax.plot(
            subset["cutoff_sec"],
            subset["converged_pct"],
            marker=marker,
            color=_temperature_gray_color(float(temperature)),
            linewidth=1.6,
            label="_nolegend_",
        )
        ax.plot(
            subset["cutoff_sec"],
            subset["compiled_pct"],
            marker=marker,
            color=_temperature_orange_color(float(temperature)),
            linewidth=1.6,
            label="_nolegend_",
        )
    ax.set_xlabel("Cutoff time (sec)")
    ax.set_ylabel("Percent of tests")
    ax.set_title("MuJoCo Compile and Convergence Rate vs. Agent Cutoff Time by Temperature")
    ax.set_ylim(0, 100)
    ax.invert_xaxis()

    # Legend 1: marker shape indicates temperature.
    temp_handles = [
        Line2D(
            [0],
            [0],
            marker=_temperature_marker(float(temperature)),
            linestyle="None",
            markersize=7,
            markerfacecolor=PRIMARY_TEXT_COLOR,
            markeredgecolor=PRIMARY_TEXT_COLOR,
            label=f"T={temperature:g}",
        )
        for temperature in temperatures
    ]
    temp_legend = ax.legend(
        handles=temp_handles,
        title="Temperature",
        frameon=True,
        ncol=4,
        fontsize=8,
        title_fontsize=9,
        loc="lower left",
        bbox_to_anchor=(0.01, 0.01),
    )
    ax.add_artist(temp_legend)

    # Legend 2: line color indicates metric.
    status_handles = [
        Line2D([0], [0], color=_temperature_orange_color(0.5), linewidth=2.0, label="Compiled"),
        Line2D([0], [0], color=_temperature_gray_color(0.5), linewidth=2.0, label="Converged"),
    ]
    ax.legend(
        handles=status_handles,
        title="Rate",
        frameon=True,
        fontsize=8,
        title_fontsize=9,
        loc="lower left",
        bbox_to_anchor=(0.33, 0.01),
    )

    _style_axis(ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def write_summary(
    output_path: Path,
    runs: list[TestRun],
    revisions: list[XmlRevision],
    global_df: pd.DataFrame,
    by_temp: pd.DataFrame,
    events_name: str,
) -> None:
    runs_with_logs = len({rev.run_id for rev in revisions})
    runs_without_logs = len(runs) - runs_with_logs
    lines = [
        "Cutoff Compile/Convergence Analysis",
        "=" * 80,
        f"Total test runs discovered: {len(runs)}",
        f"Runs with {events_name}: {runs_with_logs}",
        f"Runs without usable XML revisions: {runs_without_logs}",
        f"Total pendulum.xml revisions evaluated: {len(revisions)}",
        f"MuJoCo available: {MUJOCO_AVAILABLE}",
        "",
        "Global threshold results",
        "-" * 80,
        global_df.to_string(index=False),
        "",
        "Per-temperature final cutoff rows",
        "-" * 80,
    ]
    max_cutoff = global_df["cutoff_sec"].max() if not global_df.empty else None
    if max_cutoff is not None:
        lines.append(by_temp.loc[by_temp["cutoff_sec"].eq(max_cutoff)].to_string(index=False))
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    timing_csv = resolve_path(args.timing_csv, root) if args.timing_csv else root / "temp_testing.csv"
    analysis_csv = resolve_path(args.analysis_csv, root) if args.analysis_csv else None
    output_dir = (args.output_dir.resolve() if args.output_dir else root / "cutoff_analysis_outputs")
    plots_dir = output_dir / "cutoff_analysis_plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    if not MUJOCO_AVAILABLE:
        raise RuntimeError("MuJoCo is not installed. Install it with: pip install mujoco")

    cutoff_values = cutoffs(args.cutoff_start, args.cutoff_stop, args.cutoff_step)
    timing_lookup = load_timing_lookup(timing_csv)
    analysis_lookup = load_analysis_lookup(analysis_csv)
    runs = discover_test_runs(root, timing_lookup, analysis_lookup)
    if not runs:
        raise RuntimeError(f"No V*_Temp_*/test* folders found under {root}")

    all_revisions: list[XmlRevision] = []
    revisions_by_run: dict[str, list[XmlRevision]] = {}
    for run in runs:
        revisions = parse_revisions_for_run(
            run,
            args.events_name,
            apply_friction_fix=not args.no_friction_fix,
        )
        revisions_by_run[run.run_id] = revisions
        all_revisions.extend(revisions)

    cutoff_rows = build_cutoff_rows(runs, revisions_by_run, cutoff_values, args.include_ever_compiled)
    global_df, by_temp = summarize_thresholds(cutoff_rows, args.include_ever_compiled)
    rev_df = revision_dataframe(all_revisions)

    rev_path = output_dir / "cutoff_revision_history.csv"
    cutoff_path = output_dir / "cutoff_per_run_results.csv"
    global_path = output_dir / "cutoff_threshold_results.csv"
    by_temp_path = output_dir / "cutoff_threshold_results_by_temperature.csv"
    summary_path = output_dir / "cutoff_analysis_summary.txt"

    rev_df.to_csv(rev_path, index=False)
    cutoff_rows.to_csv(cutoff_path, index=False)
    global_df.to_csv(global_path, index=False)
    by_temp.to_csv(by_temp_path, index=False)
    write_summary(summary_path, runs, all_revisions, global_df, by_temp, args.events_name)

    plot_global_thresholds(global_df, plots_dir / "global_compile_convergence_vs_cutoff.png", args.include_ever_compiled)
    plot_metric_by_temperature(by_temp, "compiled_pct", "Compile Rate vs Cutoff Time by Temperature", plots_dir / "compile_rate_vs_cutoff_by_temperature.png", color_fn=_temperature_orange_color)
    plot_metric_by_temperature(by_temp, "converged_pct", "Convergence Rate vs Cutoff Time by Temperature", plots_dir / "convergence_rate_vs_cutoff_by_temperature.png")
    plot_combined_by_temperature(by_temp, plots_dir / "combined_compile_convergence_vs_cutoff_by_temperature.png")

    print(f"Runs discovered: {len(runs)}")
    print(f"XML revisions evaluated: {len(all_revisions)}")
    print(f"Wrote: {rev_path}")
    print(f"Wrote: {cutoff_path}")
    print(f"Wrote: {global_path}")
    print(f"Wrote: {by_temp_path}")
    print(f"Wrote: {summary_path}")
    print(f"Plots written to: {plots_dir}")


if __name__ == "__main__":
    main()
