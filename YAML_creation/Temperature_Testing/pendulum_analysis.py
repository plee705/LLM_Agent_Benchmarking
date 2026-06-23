#!/usr/bin/env python3
"""
pendulum_analysis.py

Analyzes every pendulum.xml file across temperature test folders
V2_Temp_0.1 through V9_Temp_0.5 and writes per-file pass/fail and
naming-type statistics to pendulum_analysis.csv.

Only pendulum.xml is analyzed; other .xml files in the same folder
are counted and listed but not parsed.

Run from anywhere:
    python3 Temperature_Testing/pendulum_analysis.py
"""

import csv
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path

# ── Optional MuJoCo compile check ─────────────────────────────────────────────
try:
    import mujoco as _mujoco
    MUJOCO_AVAILABLE = True
except ImportError:
    _mujoco = None
    MUJOCO_AVAILABLE = False

# ── Configuration ──────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).parent.resolve()
TEMP_PATTERN = re.compile(r'^V(\d+)_Temp_([\d.]+)$')
VERSION_RANGE = range(2, 10)          # V2 through V9 inclusive

EXPECTED_GRAVITY     = "0 -9.81 0"
EXPECTED_HINGE_AXIS  = "0 0 1"

# Top-level <mujoco> child tags that count toward extra_unexpected
EXTRA_TOP_LEVEL_TAGS = {"keyframe", "actuator", "sensor", "contact", "equality", "tendon"}

# ── CSV column order ───────────────────────────────────────────────────────────
CSV_FIELDNAMES = [
    # ── Metadata
    "temp_folder", "temperature", "test_num",
    # ── Multi-file detection
    "xml_file_count", "extra_xml_files",
    # ── Hard failures
    "xml_parse_error", "xml_parse_error_msg",
    "mujoco_compiles", "mujoco_compile_error",
    # ── Known attribute errors
    "friction_attr_error",
    # ── Deprecated compiler flags
    "deprecated_coordinate", "deprecated_angle",
    # ── Structural completeness
    "has_worldbody", "has_joint", "has_rod_geom", "has_bob_body", "has_bob_geom",
    # ── Physics / geometry correctness
    "gravity", "gravity_correct",
    "hinge_axis", "hinge_axis_correct",
    "rod_bob_pos_match",
    "rod_mass_zero", "bob_mass_nonzero",
    # ── Naming inventory
    "pendulum_body_name",
    "joint_name", "joint_type",
    "rod_geom_name", "rod_geom_type", "rod_geom_size",
    "rod_fromto", "rod_endpoint_x",
    "bob_body_name", "bob_pos", "bob_geom_name",
    # ── Extra / unexpected element count
    "extra_unexpected",
    # ── Run statistics (from temp_testing.csv)
    "elapsed_sec", "timed_out",
    "total_tokens", "input_tokens", "output_tokens",
]

TIMING_FIELDS = ("elapsed_sec", "timed_out", "total_tokens", "input_tokens", "output_tokens")

# ── Helpers ────────────────────────────────────────────────────────────────────

def _norm(s):
    """Collapse whitespace in a space-separated vector string."""
    return " ".join(s.split()) if s else None


def _floats_close(a, b, tol=1e-4):
    try:
        return abs(float(a) - float(b)) < tol
    except (ValueError, TypeError):
        return None


def _is_bob_body(body_elem):
    """True if this body contains a sphere geom as a direct child."""
    return any(
        ch.tag == "geom" and ch.get("type", "sphere") == "sphere"
        for ch in body_elem
    )


def _find_hinge_joint(body_elem):
    """Return the first joint element found anywhere inside body_elem (DFS)."""
    for ch in body_elem:
        if ch.tag == "joint":
            return ch
    for ch in body_elem:
        if ch.tag == "body":
            found = _find_hinge_joint(ch)
            if found is not None:
                return found
    return None


def _find_rod_geom(body_elem):
    """
    Return the rod geom: a non-sphere geom that has a fromto attribute.
    Recurses into nested bodies that are not the bob body.
    """
    for ch in body_elem:
        if ch.tag == "geom":
            if ch.get("type", "capsule") != "sphere" and ch.get("fromto"):
                return ch
    for ch in body_elem:
        if ch.tag == "body" and not _is_bob_body(ch):
            found = _find_rod_geom(ch)
            if found is not None:
                return found
    return None


def _find_bob_body(body_elem):
    """
    Return the bob body: a nested body that directly contains a sphere geom.
    Recurses through non-bob intermediate bodies (e.g. a 'rod' wrapper body).
    """
    for ch in body_elem:
        if ch.tag == "body":
            if _is_bob_body(ch):
                return ch
    for ch in body_elem:
        if ch.tag == "body" and not _is_bob_body(ch):
            found = _find_bob_body(ch)
            if found is not None:
                return found
    return None


# ── Per-file analysis ──────────────────────────────────────────────────────────

def _empty_row(temp_folder, temperature, test_num, test_dir):
    """Row for a test folder where pendulum.xml is absent."""
    all_xml = sorted(f.name for f in test_dir.glob("*.xml"))
    return {k: None for k in CSV_FIELDNAMES} | {
        "temp_folder":          temp_folder,
        "temperature":          temperature,
        "test_num":             test_num,
        "xml_file_count":       len(all_xml),
        "extra_xml_files":      "; ".join(all_xml) if all_xml else "",
        "xml_parse_error":      None,
        "xml_parse_error_msg":  "pendulum.xml not found",
        "mujoco_compiles":      False,
        "mujoco_compile_error": "pendulum.xml not found",
    }


def analyze_file(xml_path: Path) -> dict:
    """
    Parse and check a single pendulum.xml.
    Returns a flat dict with all CSV fields (except metadata).
    """
    r = {}

    # ── Multi-file detection ───────────────────────────────────────────────
    test_dir = xml_path.parent
    all_xml  = sorted(f.name for f in test_dir.glob("*.xml"))
    extras   = [n for n in all_xml if n != "pendulum.xml"]
    r["xml_file_count"]  = len(all_xml)
    r["extra_xml_files"] = "; ".join(extras) if extras else ""

    # ── XML parse ─────────────────────────────────────────────────────────
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError as exc:
        r["xml_parse_error"]     = True
        r["xml_parse_error_msg"] = str(exc)
        r["mujoco_compiles"]     = False
        r["mujoco_compile_error"] = str(exc)
        # Fill every remaining column with None
        for k in CSV_FIELDNAMES:
            r.setdefault(k, None)
        return r

    r["xml_parse_error"]     = False
    r["xml_parse_error_msg"] = None

    # ── MuJoCo compile check ───────────────────────────────────────────────
    if MUJOCO_AVAILABLE:
        try:
            _mujoco.MjModel.from_xml_path(str(xml_path))
            r["mujoco_compiles"]      = True
            r["mujoco_compile_error"] = None
        except Exception as exc:
            r["mujoco_compiles"]      = False
            r["mujoco_compile_error"] = " ".join(str(exc).split())  # collapse newlines
    else:
        r["mujoco_compiles"]      = None   # unavailable — not a fail
        r["mujoco_compile_error"] = None

    # ── Compiler flags ────────────────────────────────────────────────────
    compiler = root.find("compiler")
    r["deprecated_coordinate"] = (
        compiler is not None and "coordinate" in compiler.attrib
    )
    r["deprecated_angle"] = (
        compiler is not None and "angle" in compiler.attrib
    )

    # ── Option / gravity ─────────────────────────────────────────────────
    option = root.find("option")
    gravity_raw  = option.get("gravity") if option is not None else None
    r["gravity"]         = _norm(gravity_raw)
    r["gravity_correct"] = (r["gravity"] == EXPECTED_GRAVITY)

    # ── Worldbody ─────────────────────────────────────────────────────────
    worldbody = root.find("worldbody")
    r["has_worldbody"] = worldbody is not None

    # ── Extra / unexpected element count ──────────────────────────────────
    extra_count = 0
    for ch in root:
        if ch.tag in EXTRA_TOP_LEVEL_TAGS:
            extra_count += 1
    # World-level geoms (direct geom children of worldbody, e.g. ground plane)
    if worldbody is not None:
        for ch in worldbody:
            if ch.tag == "geom":
                extra_count += 1
    r["extra_unexpected"] = extra_count

    # ── Nothing more possible without worldbody ───────────────────────────
    if worldbody is None:
        for k in CSV_FIELDNAMES:
            r.setdefault(k, None)
        return r

    # ── Main pendulum body ────────────────────────────────────────────────
    pend_bodies = [ch for ch in worldbody if ch.tag == "body"]
    if not pend_bodies:
        for k in CSV_FIELDNAMES:
            r.setdefault(k, None)
        return r

    pend_body = pend_bodies[0]
    r["pendulum_body_name"] = pend_body.get("name", "unnamed")

    # ── Joint ─────────────────────────────────────────────────────────────
    joint = _find_hinge_joint(pend_body)
    r["has_joint"] = joint is not None
    if joint is not None:
        r["joint_name"]    = joint.get("name", "unnamed")
        r["joint_type"]    = joint.get("type", "hinge")
        hinge_axis_raw     = joint.get("axis")
        r["hinge_axis"]    = _norm(hinge_axis_raw)
        r["hinge_axis_correct"] = (r["hinge_axis"] == EXPECTED_HINGE_AXIS)
        # friction= is wrong; frictionloss= is correct
        r["friction_attr_error"] = (
            "friction" in joint.attrib and "frictionloss" not in joint.attrib
        )
    else:
        r["joint_name"]          = None
        r["joint_type"]          = None
        r["hinge_axis"]          = None
        r["hinge_axis_correct"]  = None
        r["friction_attr_error"] = None

    # ── Rod geom ──────────────────────────────────────────────────────────
    rod = _find_rod_geom(pend_body)
    r["has_rod_geom"]   = rod is not None
    if rod is not None:
        r["rod_geom_name"] = rod.get("name", "unnamed")
        r["rod_geom_type"] = rod.get("type", "capsule")
        r["rod_geom_size"] = rod.get("size")
        rod_mass_raw       = rod.get("mass")
        r["rod_mass_zero"] = _floats_close(rod_mass_raw, 0.0) if rod_mass_raw is not None else None
        fromto             = rod.get("fromto")
        r["rod_fromto"]    = fromto
        if fromto:
            parts = fromto.split()
            r["rod_endpoint_x"] = parts[3] if len(parts) >= 6 else None
        else:
            r["rod_endpoint_x"] = None
    else:
        r["rod_geom_name"]  = None
        r["rod_geom_type"]  = None
        r["rod_geom_size"]  = None
        r["rod_mass_zero"]  = None
        r["rod_fromto"]     = None
        r["rod_endpoint_x"] = None

    # ── Bob body & geom ───────────────────────────────────────────────────
    bob_body = _find_bob_body(pend_body)
    r["has_bob_body"] = bob_body is not None
    if bob_body is not None:
        r["bob_body_name"] = bob_body.get("name", "unnamed")
        bob_pos_raw        = bob_body.get("pos")
        r["bob_pos"]       = _norm(bob_pos_raw)
        bob_x              = bob_pos_raw.split()[0] if bob_pos_raw else None

        # Rod endpoint vs bob position match
        if r["rod_endpoint_x"] is not None and bob_x is not None:
            r["rod_bob_pos_match"] = _floats_close(r["rod_endpoint_x"], bob_x)
        else:
            r["rod_bob_pos_match"] = None

        bob_geom = bob_body.find("geom")
        r["has_bob_geom"] = bob_geom is not None
        if bob_geom is not None:
            r["bob_geom_name"]    = bob_geom.get("name", "unnamed")
            bob_mass_raw          = bob_geom.get("mass")
            r["bob_mass_nonzero"] = (
                not _floats_close(bob_mass_raw, 0.0)
                if bob_mass_raw is not None else None
            )
        else:
            r["bob_geom_name"]    = None
            r["bob_mass_nonzero"] = None
    else:
        r["bob_body_name"]   = None
        r["bob_pos"]         = None
        r["rod_bob_pos_match"] = None
        r["has_bob_geom"]    = False
        r["bob_geom_name"]   = None
        r["bob_mass_nonzero"] = None

    # Fill any remaining fields with None (safety net)
    for k in CSV_FIELDNAMES:
        r.setdefault(k, None)

    return r


# ── temp_testing.csv loader ──────────────────────────────────────────────────

def _load_timing(csv_path: Path) -> dict:
    """
    Load run statistics from temp_testing.csv.
    Returns a dict keyed by (temp_folder, test_num) -> {field: value}.
    run_id format: "V2_Temp_0.1/test01"  →  temp_folder="V2_Temp_0.1", test_num="test01"
    """
    lookup = {}
    if not csv_path.exists():
        return lookup
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            run_id = row.get("run_id", "")
            if "/" not in run_id:
                continue
            temp_folder, test_num = run_id.split("/", 1)
            lookup[(temp_folder, test_num)] = {f: row.get(f) for f in TIMING_FIELDS}
    return lookup


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    rows = []
    timing = _load_timing(SCRIPT_DIR / "temp_testing.csv")

    for vnum in VERSION_RANGE:
        for temp_dir in sorted(SCRIPT_DIR.glob(f"V{vnum}_Temp_*")):
            if not temp_dir.is_dir():
                continue
            m = TEMP_PATTERN.match(temp_dir.name)
            if not m:
                continue
            temperature = float(m.group(2))

            for test_dir in sorted(temp_dir.glob("test*")):
                if not test_dir.is_dir():
                    continue
                test_num = test_dir.name
                xml_path = test_dir / "pendulum.xml"

                if not xml_path.exists():
                    row = _empty_row(temp_dir.name, temperature, test_num, test_dir)
                else:
                    result = analyze_file(xml_path)
                    row = {
                        **result,
                        "temp_folder": temp_dir.name,
                        "temperature": temperature,
                        "test_num":    test_num,
                    }

                # ── Merge timing / token stats ──────────────────────────────
                stats = timing.get((temp_dir.name, test_num), {})
                for f in TIMING_FIELDS:
                    row[f] = stats.get(f)  # None if run_id not found in timing CSV

                rows.append(row)

    # ── Write CSV with versioning ─────────────────────────────────────────
    # Find the next available version number
    existing = list(SCRIPT_DIR.glob("pendulum_analysis[0-9]*.csv"))
    version = 1
    if existing:
        # Extract version numbers from filenames like pendulum_analysis1.csv
        versions = []
        for f in existing:
            match = re.search(r'pendulum_analysis(\d+)\.csv$', f.name)
            if match:
                versions.append(int(match.group(1)))
        if versions:
            version = max(versions) + 1
    
    out_path = SCRIPT_DIR / f"pendulum_analysis{version}.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    # ── Console summary ────────────────────────────────────────────────────
    total   = len(rows)
    found   = sum(1 for r in rows if r.get("xml_parse_error_msg") != "pendulum.xml not found")
    missing = total - found
    parse_err   = sum(1 for r in rows if r.get("xml_parse_error") is True)
    friction_err = sum(1 for r in rows if r.get("friction_attr_error") is True)
    multi_xml   = sum(1 for r in rows if (r.get("xml_file_count") or 0) > 1)

    print(f"\n{'='*55}")
    print(f"  pendulum_analysis.py  —  {total} test folders scanned")
    print(f"{'='*55}")
    print(f"  pendulum.xml found    : {found:>4}  ({missing} missing)")
    print(f"  XML parse errors      : {parse_err:>4}")
    print(f"  Multi-XML folders     : {multi_xml:>4}  (have extra .xml files)")
    print(f"  friction= attr error  : {friction_err:>4}  / {found}")

    if MUJOCO_AVAILABLE:
        compiles   = sum(1 for r in rows if r.get("mujoco_compiles") is True)
        fails      = sum(1 for r in rows if r.get("mujoco_compiles") is False
                         and r.get("xml_parse_error_msg") != "pendulum.xml not found")
        pct = 100 * compiles / found if found else 0
        print(f"  MuJoCo compile pass   : {compiles:>4}  / {found}  ({pct:.1f}%)")
        print(f"  MuJoCo compile fail   : {fails:>4}  / {found}")
    else:
        print("  MuJoCo not installed  : compile check skipped (install with: pip install mujoco)")

    print(f"\n  Output: {out_path}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
