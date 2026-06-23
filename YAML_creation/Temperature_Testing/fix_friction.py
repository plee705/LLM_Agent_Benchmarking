#!/usr/bin/env python3
"""
Fix malformed MuJoCo joint friction attributes in temperature test XML files.

This script scans pendulum.xml files under V2_Temp_0.1 through V9_Temp_0.5
and replaces the invalid attribute `friction="0"` with `frictionloss="0"`.

Run from anywhere:
    python3 Temperature_Testing/fix_friction.py
"""

from pathlib import Path


SCRIPT_DIR = Path(__file__).parent.resolve()
TARGET_DIRS = (
    "V2_Temp_0.1",
    "V3_Temp_0.4",
    "V4_Temp_0.7",
    "V5_Temp_1.0",
    "V6_Temp_0.0",
    "V7_Temp_0.2",
    "V8_Temp_0.3",
    "V9_Temp_0.5",
)
OLD_ATTR = 'friction="0"'
NEW_ATTR = 'frictionloss="0"'


def iter_pendulum_files():
    for folder_name in TARGET_DIRS:
        root = SCRIPT_DIR / folder_name
        if not root.is_dir():
            continue
        yield from root.rglob("pendulum.xml")


def main():
    files_updated = 0
    replacements = 0

    for xml_path in sorted(iter_pendulum_files()):
        text = xml_path.read_text(encoding="utf-8")
        count = text.count(OLD_ATTR)
        if count == 0:
            continue

        xml_path.write_text(text.replace(OLD_ATTR, NEW_ATTR), encoding="utf-8")
        files_updated += 1
        replacements += count
        print(f"updated {xml_path.relative_to(SCRIPT_DIR)} ({count} replacement{'s' if count != 1 else ''})")

    print(f"done: updated {files_updated} file(s), {replacements} replacement(s)")


if __name__ == "__main__":
    main()