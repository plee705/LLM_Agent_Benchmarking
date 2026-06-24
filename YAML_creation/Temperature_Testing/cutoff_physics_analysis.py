#!/usr/bin/env python3
"""
cutoff_physics_analysis.py

Reconstructs pendulum.xml revisions from OpenCode JSONL event logs and evaluates
how physics correctness changes as the hypothetical timeout cutoff is reduced.

This follows the project layout and cutoff method used by
cutoff_compile_convergence_analysis.py:

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

For each cutoff, the latest pendulum.xml write event before that cutoff is used.
The XML is compiled, simulated in MuJoCo, and evaluated with MuJoCo's built-in
energy calculations. A model that fails to compile fails every physics category.

Outputs:
    cutoff_physics_revision_history.csv
    cutoff_pend_physics_test.csv
    cutoff_physics_threshold_results.csv
    cutoff_physics_threshold_results_by_temperature.csv
    cutoff_physics_analysis_summary.txt
    cutoff_physics_analysis_plots/*.png

Run:
    python3 cutoff_physics_analysis.py
    python3 cutoff_physics_analysis.py --root /path/to/Temperature_Testing
    python3 cutoff_physics_analysis.py --analysis-csv pendulum_analysis_fixedfriction.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

try:
    import mujoco
    MUJOCO_AVAILABLE = True
except ImportError:
    mujoco = None
    MUJOCO_AVAILABLE = False

TEMP_PATTERN = re.compile(r"^V(\d+)_Temp_([\d.]+)$")
DEFAULT_CUTOFF_START = 150
DEFAULT_CUTOFF_STOP = 0
DEFAULT_CUTOFF_STEP = 10

TARGET_FREQUENCY_HZ = 1.0
TARGET_ENERGY_J = 2.4378

PRIMARY_TEXT_COLOR = "#4D4D4F"
SECONDARY_COLOR = "#BFBFBF"
FREQUENCY_COLOR = "#324458"
ENERGY_COLOR = "#F47B20"
TOTAL_COLOR = "#868687"


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
class XmlSnapshot:
    temp_folder: str
    temperature: float
    test_num: str
    run_id: str
    revision_index: int
    elapsed_sec: float
    timestamp_raw: Any
    xml_text: str
    source_file: str


@dataclass(frozen=True)
class PhysicsResult:
    temp_folder: str
    temperature: float
    test_num: str
    run_id: str
    revision_index: int
    revision_elapsed_sec: float
    compiles: bool
    compile_error: str
    simulated: bool
    simulation_error: str
    hinge_joint_name: str
    period_sec: float | None
    frequency_hz: float | None
    initial_potential_energy_j: float | None
    min_potential_energy_j: float | None
    potential_energy_drop_j: float | None
    max_kinetic_energy_j: float | None
    initial_total_energy_j: float | None
    total_energy_range_j: float | None
    total_energy_range_pct: float | None
    frequency_pass: bool
    potential_energy_pass: bool
    kinetic_energy_pass: bool
    total_energy_constant_pass: bool
    energy_pass: bool
    physics_pass: bool
    samples: int
    xml_chars: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--timing-csv", type=Path, default=None)
    parser.add_argument("--analysis-csv", type=Path, default=None)
    parser.add_argument("--events-name", default="opencode_events.jsonl")
    parser.add_argument("--cutoff-start", type=int, default=DEFAULT_CUTOFF_START)
    parser.add_argument("--cutoff-stop", type=int, default=DEFAULT_CUTOFF_STOP)
    parser.add_argument("--cutoff-step", type=int, default=DEFAULT_CUTOFF_STEP)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--timestep", type=float, default=0.01)
    parser.add_argument("--max-sim-time", type=float, default=5.0)
    parser.add_argument("--frequency-tol", type=float, default=0.01)
    parser.add_argument("--energy-tol", type=float, default=0.05)
    parser.add_argument("--total-energy-range-tol-pct", type=float, default=2.0)
    parser.add_argument("--no-friction-fix", action="store_true")
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
    match = re.search(r"test0*(\d+)$", text)
    return f"test{int(match.group(1))}" if match else text


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
    return candidate if candidate.exists() else path


def load_timing_lookup(csv_path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    if not csv_path.exists():
        return lookup
    with csv_path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            run_id = str(row.get("run_id", "")).strip()
            if "/" in run_id:
                temp_folder, test_num = run_id.split("/", 1)
            else:
                temp_folder = str(row.get("temp_folder", "")).strip()
                test_num = str(row.get("test_num", "")).strip()
            if not temp_folder or not test_num:
                continue
            record = {
                "run_id": run_id or f"{temp_folder}/{test_num}",
                "elapsed_sec": to_float(row.get("elapsed_sec")),
                "timed_out": normalize_bool(row.get("timed_out")),
                "temperature": to_float(row.get("temp", row.get("temperature"))),
            }
            for key in possible_test_keys(temp_folder, test_num):
                lookup[key] = record
    return lookup


def load_analysis_lookup(csv_path: Path | None) -> dict[tuple[str, str], dict[str, Any]]:
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    if csv_path is None or not csv_path.exists():
        return lookup
    with csv_path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
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
            metadata = None
            for key in possible_test_keys(temp_dir.name, test_dir.name):
                metadata = timing_lookup.get(key) or analysis_lookup.get(key)
                if metadata:
                    break
            runs.append(
                TestRun(
                    temp_folder=temp_dir.name,
                    temperature=float(metadata.get("temperature") if metadata and metadata.get("temperature") is not None else folder_temp),
                    test_num=test_dir.name,
                    test_dir=test_dir,
                    run_id=metadata.get("run_id") if metadata and metadata.get("run_id") else f"{temp_dir.name}/{test_dir.name}",
                    elapsed_sec=metadata.get("elapsed_sec") if metadata else None,
                    timed_out=metadata.get("timed_out") if metadata else None,
                )
            )
    return runs


def timestamp_to_seconds(timestamp: Any) -> float | None:
    if timestamp is None:
        return None
    if isinstance(timestamp, (int, float)):
        value = float(timestamp)
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
        return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def is_pendulum_xml_path(path_value: Any) -> bool:
    return path_value is not None and Path(str(path_value)).name == "pendulum.xml"


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
    return (ET.tostring(root, encoding="unicode"), fixes) if fixes else (xml_text, 0)


def snapshots_for_run(run: TestRun, events_name: str) -> list[XmlSnapshot]:
    events_path = run.test_dir / events_name
    if not events_path.exists():
        return []
    raw: list[tuple[float, Any, str]] = []
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
                state = ((event.get("part") or {}).get("state") or {})
                ts_sec = timestamp_to_seconds(((state.get("time") or {}).get("start")))
            if ts_sec is None:
                ts_sec = float(line_number)
            raw.append((ts_sec, event.get("timestamp"), xml_text))
    if not raw:
        return []
    raw.sort(key=lambda item: item[0])
    start_ts = raw[0][0]
    return [
        XmlSnapshot(
            temp_folder=run.temp_folder,
            temperature=run.temperature,
            test_num=run.test_num,
            run_id=run.run_id,
            revision_index=idx,
            elapsed_sec=max(0.0, ts_sec - start_ts),
            timestamp_raw=timestamp_raw,
            xml_text=xml_text,
            source_file=str(events_path),
        )
        for idx, (ts_sec, timestamp_raw, xml_text) in enumerate(raw, start=1)
    ]


def cutoffs(start: int, stop: int, step: int) -> list[int]:
    if step <= 0:
        raise ValueError("cutoff-step must be positive")
    return list(range(start, stop - 1, -step)) if start >= stop else list(range(start, stop + 1, step))


def latest_snapshot_before(snapshots: list[XmlSnapshot], cutoff: float) -> XmlSnapshot | None:
    eligible = [snap for snap in snapshots if snap.elapsed_sec <= cutoff]
    return eligible[-1] if eligible else None


def first_hinge_joint(model: Any) -> tuple[int | None, str]:
    for joint_id in range(model.njnt):
        if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_HINGE:
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) or f"joint_{joint_id}"
            return joint_id, name
    return None, ""


def sign_no_zero(value: float) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def interpolate_crossing_time(t0: float, y0: float, t1: float, y1: float) -> float:
    denom = y1 - y0
    if abs(denom) < 1.0e-12:
        return t1
    return t0 - y0 * (t1 - t0) / denom


def simulate_physics(
    snapshot: XmlSnapshot,
    timestep: float,
    max_sim_time: float,
    frequency_tol: float,
    energy_tol: float,
    total_energy_range_tol_pct: float,
    apply_friction_fix: bool,
) -> PhysicsResult:
    xml_text = snapshot.xml_text
    fix_count = 0
    if apply_friction_fix:
        xml_text, fix_count = apply_prompt_error_friction_fix(xml_text)
    try:
        ET.fromstring(xml_text)
    except ET.ParseError as exc:
        return failed_result(snapshot, False, f"XML parse error: {' '.join(str(exc).split())}", len(xml_text))
    if not MUJOCO_AVAILABLE:
        return failed_result(snapshot, False, "MuJoCo not installed; install with: pip install mujoco", len(xml_text))

    with tempfile.TemporaryDirectory() as tmp:
        xml_path = Path(tmp) / "pendulum.xml"
        xml_path.write_text(xml_text, encoding="utf-8")
        try:
            model = mujoco.MjModel.from_xml_path(str(xml_path))
        except Exception as exc:
            msg = " ".join(str(exc).split())
            if fix_count:
                msg = f"friction_fix_applied={fix_count}; {msg}"
            return failed_result(snapshot, False, msg, len(xml_text))

    model.opt.timestep = timestep
    model.opt.enableflags |= int(mujoco.mjtEnableBit.mjENBL_ENERGY)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    joint_id, joint_name = first_hinge_joint(model)
    if joint_id is None:
        return failed_result(snapshot, True, "", len(xml_text), simulated=False, simulation_error="No hinge joint found")
    qpos_i = int(model.jnt_qposadr[joint_id])
    qvel_i = int(model.jnt_dofadr[joint_id])

    times = [0.0]
    qvels = [float(data.qvel[qvel_i])]
    potentials = [float(data.energy[0])]
    kinetics = [float(data.energy[1])]
    totals = [float(data.energy[0] + data.energy[1])]

    initial_motion_sign = 0
    completed_period = None
    min_detection_time = 0.25

    n_steps = int(math.ceil(max_sim_time / timestep))
    for step in range(1, n_steps + 1):
        mujoco.mj_step(model, data)
        t = step * timestep
        qvel = float(data.qvel[qvel_i])
        times.append(t)
        qvels.append(qvel)
        potentials.append(float(data.energy[0]))
        kinetics.append(float(data.energy[1]))
        totals.append(float(data.energy[0] + data.energy[1]))

        s = sign_no_zero(qvel)
        if initial_motion_sign == 0 and s != 0:
            initial_motion_sign = s

        prev_qvel = qvels[-2]
        prev_s = sign_no_zero(prev_qvel)
        if initial_motion_sign and prev_s == -initial_motion_sign and s == initial_motion_sign and t >= min_detection_time:
            completed_period = interpolate_crossing_time(times[-2], prev_qvel, t, qvel)
            break

    if completed_period is None or completed_period <= 0:
        return failed_result(
            snapshot,
            True,
            "" if not fix_count else f"friction_fix_applied={fix_count}",
            len(xml_text),
            simulated=False,
            simulation_error="One full oscillation was not detected before max simulation time",
            hinge_joint_name=joint_name,
            samples=len(times),
        )

    frequency = 1.0 / completed_period
    initial_pe = potentials[0]
    min_pe = min(potentials)
    pe_drop = initial_pe - min_pe
    max_ke = max(kinetics)
    initial_total = totals[0]
    total_range = max(totals) - min(totals)
    total_range_pct = 100.0 * total_range / abs(initial_total) if abs(initial_total) > 1.0e-12 else None

    frequency_pass = abs(frequency - TARGET_FREQUENCY_HZ) <= frequency_tol
    potential_pass = abs(pe_drop - TARGET_ENERGY_J) <= energy_tol
    kinetic_pass = abs(max_ke - TARGET_ENERGY_J) <= energy_tol
    total_pass = total_range_pct is not None and total_range_pct <= total_energy_range_tol_pct
    energy_pass = potential_pass and kinetic_pass and total_pass

    return PhysicsResult(
        temp_folder=snapshot.temp_folder,
        temperature=snapshot.temperature,
        test_num=snapshot.test_num,
        run_id=snapshot.run_id,
        revision_index=snapshot.revision_index,
        revision_elapsed_sec=snapshot.elapsed_sec,
        compiles=True,
        compile_error="" if not fix_count else f"friction_fix_applied={fix_count}",
        simulated=True,
        simulation_error="",
        hinge_joint_name=joint_name,
        period_sec=completed_period,
        frequency_hz=frequency,
        initial_potential_energy_j=initial_pe,
        min_potential_energy_j=min_pe,
        potential_energy_drop_j=pe_drop,
        max_kinetic_energy_j=max_ke,
        initial_total_energy_j=initial_total,
        total_energy_range_j=total_range,
        total_energy_range_pct=total_range_pct,
        frequency_pass=frequency_pass,
        potential_energy_pass=potential_pass,
        kinetic_energy_pass=kinetic_pass,
        total_energy_constant_pass=total_pass,
        energy_pass=energy_pass,
        physics_pass=frequency_pass and energy_pass,
        samples=len(times),
        xml_chars=len(xml_text),
    )


def failed_result(
    snapshot: XmlSnapshot,
    compiles: bool,
    compile_error: str,
    xml_chars: int,
    simulated: bool = False,
    simulation_error: str = "",
    hinge_joint_name: str = "",
    samples: int = 0,
) -> PhysicsResult:
    return PhysicsResult(
        temp_folder=snapshot.temp_folder,
        temperature=snapshot.temperature,
        test_num=snapshot.test_num,
        run_id=snapshot.run_id,
        revision_index=snapshot.revision_index,
        revision_elapsed_sec=snapshot.elapsed_sec,
        compiles=compiles,
        compile_error=compile_error,
        simulated=simulated,
        simulation_error=simulation_error,
        hinge_joint_name=hinge_joint_name,
        period_sec=None,
        frequency_hz=None,
        initial_potential_energy_j=None,
        min_potential_energy_j=None,
        potential_energy_drop_j=None,
        max_kinetic_energy_j=None,
        initial_total_energy_j=None,
        total_energy_range_j=None,
        total_energy_range_pct=None,
        frequency_pass=False,
        potential_energy_pass=False,
        kinetic_energy_pass=False,
        total_energy_constant_pass=False,
        energy_pass=False,
        physics_pass=False,
        samples=samples,
        xml_chars=xml_chars,
    )


def build_cutoff_rows(
    runs: list[TestRun],
    snapshots_by_run: dict[str, list[XmlSnapshot]],
    physics_by_snapshot: dict[tuple[str, int], PhysicsResult],
    cutoff_values: list[int],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run in runs:
        snapshots = snapshots_by_run.get(run.run_id, [])
        for cutoff in cutoff_values:
            snap = latest_snapshot_before(snapshots, cutoff)
            if snap is None:
                row = {
                    "temp_folder": run.temp_folder,
                    "temperature": run.temperature,
                    "test_num": run.test_num,
                    "run_id": run.run_id,
                    "cutoff_sec": cutoff,
                    "revision_used": None,
                    "revision_elapsed_sec": None,
                    "compiled_at_cutoff": False,
                    "frequency_pass": False,
                    "energy_pass": False,
                    "physics_pass": False,
                    "compile_error_at_cutoff": "No pendulum.xml revision before cutoff",
                    "simulation_error_at_cutoff": "",
                }
            else:
                result = physics_by_snapshot[(snap.run_id, snap.revision_index)]
                row = asdict(result)
                row.update(
                    {
                        "cutoff_sec": cutoff,
                        "revision_used": snap.revision_index,
                        "compiled_at_cutoff": result.compiles,
                        "compile_error_at_cutoff": result.compile_error,
                        "simulation_error_at_cutoff": result.simulation_error,
                    }
                )
            rows.append(row)
    return pd.DataFrame.from_records(rows)


def summarize_thresholds(cutoff_rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    agg = {
        "tests": ("run_id", "count"),
        "compiled_count": ("compiled_at_cutoff", "sum"),
        "frequency_pass_count": ("frequency_pass", "sum"),
        "energy_pass_count": ("energy_pass", "sum"),
        "physics_pass_count": ("physics_pass", "sum"),
    }
    global_df = cutoff_rows.groupby("cutoff_sec", as_index=False).agg(**agg)
    by_temp = cutoff_rows.groupby(["temperature", "cutoff_sec"], as_index=False).agg(**agg)
    for df in (global_df, by_temp):
        df["compiled_pct"] = 100.0 * df["compiled_count"] / df["tests"]
        df["frequency_pass_pct"] = 100.0 * df["frequency_pass_count"] / df["tests"]
        df["energy_pass_pct"] = 100.0 * df["energy_pass_count"] / df["tests"]
        df["physics_pass_pct"] = 100.0 * df["physics_pass_count"] / df["tests"]
    return (
        global_df.sort_values("cutoff_sec", ascending=False),
        by_temp.sort_values(["temperature", "cutoff_sec"], ascending=[True, False]),
    )


def _style_axis(ax: plt.Axes) -> None:
    for spine in ax.spines.values():
        spine.set_color(PRIMARY_TEXT_COLOR)
    ax.tick_params(colors=PRIMARY_TEXT_COLOR)
    ax.xaxis.label.set_color(PRIMARY_TEXT_COLOR)
    ax.yaxis.label.set_color(PRIMARY_TEXT_COLOR)
    ax.title.set_color(PRIMARY_TEXT_COLOR)
    ax.grid(axis="y", linestyle=":", color=SECONDARY_COLOR, alpha=0.9)


def _temperature_marker(temp: float) -> str:
    marker_map = {0.0: "o", 0.1: "s", 0.2: "^", 0.3: "v", 0.4: "D", 0.5: "*", 0.7: "p", 1.0: "H"}
    return marker_map.get(float(temp), "o")


def plot_combined_by_temperature(by_temp: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))
    temperatures = sorted(by_temp["temperature"].dropna().unique())
    for temperature in temperatures:
        subset = by_temp.loc[by_temp["temperature"].eq(temperature)].sort_values("cutoff_sec", ascending=False)
        marker = _temperature_marker(float(temperature))
        ax.plot(subset["cutoff_sec"], subset["frequency_pass_pct"], marker=marker, color=FREQUENCY_COLOR, linewidth=1.5, alpha=0.75)
        ax.plot(subset["cutoff_sec"], subset["energy_pass_pct"], marker=marker, color=ENERGY_COLOR, linewidth=1.5, alpha=0.75)
    ax.set_xlabel("Cutoff time (sec)")
    ax.set_ylabel("Percent of tests")
    ax.set_title("Pendulum Physics Pass Rate vs. Agent Cutoff Time by Temperature")
    ax.set_ylim(0, 100)
    ax.invert_xaxis()

    temp_handles = [
        Line2D([0], [0], marker=_temperature_marker(float(t)), linestyle="None", markersize=7,
               markerfacecolor=PRIMARY_TEXT_COLOR, markeredgecolor=PRIMARY_TEXT_COLOR, label=f"T={t:g}")
        for t in temperatures
    ]
    temp_legend = ax.legend(handles=temp_handles, title="Temperature", frameon=True, ncol=4, fontsize=8,
                            title_fontsize=9, loc="lower left", bbox_to_anchor=(0.01, 0.01))
    ax.add_artist(temp_legend)
    metric_handles = [
        Line2D([0], [0], color=FREQUENCY_COLOR, linewidth=2.0, label="Frequency pass"),
        Line2D([0], [0], color=ENERGY_COLOR, linewidth=2.0, label="Energy pass"),
    ]
    ax.legend(handles=metric_handles, title="Check", frameon=True, fontsize=8, title_fontsize=9,
              loc="lower left", bbox_to_anchor=(0.35, 0.01))
    _style_axis(ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_global(global_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(global_df["cutoff_sec"], global_df["frequency_pass_pct"], marker="o", color=FREQUENCY_COLOR, label="Frequency pass")
    ax.plot(global_df["cutoff_sec"], global_df["energy_pass_pct"], marker="s", color=ENERGY_COLOR, label="Energy pass")
    ax.plot(global_df["cutoff_sec"], global_df["physics_pass_pct"], marker="^", color=TOTAL_COLOR, label="Both pass")
    ax.set_xlabel("Cutoff time (sec)")
    ax.set_ylabel("Percent of tests")
    ax.set_title("Pendulum Physics Pass Rate vs Cutoff Time")
    ax.set_ylim(0, 100)
    ax.invert_xaxis()
    ax.legend(frameon=False)
    _style_axis(ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def write_summary(path: Path, runs: list[TestRun], snapshots: list[XmlSnapshot], global_df: pd.DataFrame, by_temp: pd.DataFrame, args: argparse.Namespace) -> None:
    lines = [
        "Cutoff Pendulum Physics Analysis",
        "=" * 80,
        f"Total test runs discovered: {len(runs)}",
        f"Total pendulum.xml revisions discovered: {len(snapshots)}",
        f"MuJoCo available: {MUJOCO_AVAILABLE}",
        f"Timestep: {args.timestep:g} sec",
        f"Target frequency: {TARGET_FREQUENCY_HZ:g} Hz +/- {args.frequency_tol:g} Hz",
        f"Target energy transfer: {TARGET_ENERGY_J:g} J +/- {args.energy_tol:g} J",
        f"Total energy range tolerance: {args.total_energy_range_tol_pct:g}%",
        "",
        "Global threshold results",
        "-" * 80,
        global_df.to_string(index=False),
        "",
        "Per-temperature rows at maximum cutoff",
        "-" * 80,
    ]
    if not global_df.empty:
        max_cutoff = global_df["cutoff_sec"].max()
        lines.append(by_temp.loc[by_temp["cutoff_sec"].eq(max_cutoff)].to_string(index=False))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    timing_csv = resolve_path(args.timing_csv, root) if args.timing_csv else root / "temp_testing.csv"
    analysis_csv = resolve_path(args.analysis_csv, root) if args.analysis_csv else None
    output_dir = args.output_dir.resolve() if args.output_dir else root / "cutoff_physics_analysis_outputs"
    plots_dir = output_dir / "cutoff_physics_analysis_plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    if not MUJOCO_AVAILABLE:
        raise RuntimeError("MuJoCo is not installed. Install it with: pip install mujoco")

    runs = discover_test_runs(root, load_timing_lookup(timing_csv), load_analysis_lookup(analysis_csv))
    if not runs:
        raise RuntimeError(f"No V*_Temp_*/test* folders found under {root}")

    snapshots_by_run: dict[str, list[XmlSnapshot]] = {}
    all_snapshots: list[XmlSnapshot] = []
    for run in runs:
        snapshots = snapshots_for_run(run, args.events_name)
        snapshots_by_run[run.run_id] = snapshots
        all_snapshots.extend(snapshots)

    physics_by_snapshot: dict[tuple[str, int], PhysicsResult] = {}
    for snapshot in all_snapshots:
        physics_by_snapshot[(snapshot.run_id, snapshot.revision_index)] = simulate_physics(
            snapshot=snapshot,
            timestep=args.timestep,
            max_sim_time=args.max_sim_time,
            frequency_tol=args.frequency_tol,
            energy_tol=args.energy_tol,
            total_energy_range_tol_pct=args.total_energy_range_tol_pct,
            apply_friction_fix=not args.no_friction_fix,
        )

    cutoff_rows = build_cutoff_rows(runs, snapshots_by_run, physics_by_snapshot, cutoffs(args.cutoff_start, args.cutoff_stop, args.cutoff_step))
    global_df, by_temp = summarize_thresholds(cutoff_rows)
    revision_df = pd.DataFrame.from_records([asdict(r) for r in physics_by_snapshot.values()])

    revision_path = output_dir / "cutoff_physics_revision_history.csv"
    cutoff_path = output_dir / "cutoff_pend_physics_test.csv"
    global_path = output_dir / "cutoff_physics_threshold_results.csv"
    by_temp_path = output_dir / "cutoff_physics_threshold_results_by_temperature.csv"
    summary_path = output_dir / "cutoff_physics_analysis_summary.txt"

    revision_df.to_csv(revision_path, index=False)
    cutoff_rows.to_csv(cutoff_path, index=False)
    global_df.to_csv(global_path, index=False)
    by_temp.to_csv(by_temp_path, index=False)
    write_summary(summary_path, runs, all_snapshots, global_df, by_temp, args)

    plot_global(global_df, plots_dir / "global_physics_vs_cutoff.png")
    plot_combined_by_temperature(by_temp, plots_dir / "combined_physics_vs_cutoff_by_temperature.png")

    print(f"Runs discovered: {len(runs)}")
    print(f"XML revisions evaluated: {len(all_snapshots)}")
    print(f"Wrote: {revision_path}")
    print(f"Wrote: {cutoff_path}")
    print(f"Wrote: {global_path}")
    print(f"Wrote: {by_temp_path}")
    print(f"Wrote: {summary_path}")
    print(f"Plots written to: {plots_dir}")


if __name__ == "__main__":
    main()
