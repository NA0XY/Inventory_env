#!/usr/bin/env python3
"""Validate strict START/STEP/END inference log compliance.

Checks:
- stdout contains only [START], [STEP], [END] lines matching required formats
- stderr contains no structured [START]/[STEP]/[END] lines
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


START_RX = re.compile(r"^\[START\] task=\S+ env=\S+ model=.+$")
STEP_RX = re.compile(
    r"^\[STEP\] step=\d+ action=.+ reward=-?\d+\.\d{2} done=(true|false) error=.*$"
)
END_RX = re.compile(r"^\[END\] success=(true|false) steps=\d+ score=-?\d+\.\d{2} rewards=.*$")
STRUCTURED_RX = re.compile(r"^\[(START|STEP|END)\] ")


def read_lines(path: Path) -> list[str]:
    data = path.read_bytes()

    if data.startswith(b"\xff\xfe") or data.startswith(b"\xfe\xff"):
        # Use utf-16 (not endian-specific) so BOM is consumed.
        text = data.decode("utf-16")
    elif data.startswith(b"\xef\xbb\xbf"):
        text = data.decode("utf-8-sig")
    else:
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            text = data.decode("utf-16")

    return [line.rstrip("\r\n") for line in text.splitlines()]


def validate_stdout(lines: list[str]) -> list[str]:
    bad_lines: list[str] = []

    for line in lines:
        if not line:
            continue
        if line.startswith("[START] "):
            if not START_RX.match(line):
                bad_lines.append(line)
            continue
        if line.startswith("[STEP] "):
            if not STEP_RX.match(line):
                bad_lines.append(line)
            continue
        if line.startswith("[END] "):
            if not END_RX.match(line):
                bad_lines.append(line)
            continue
        bad_lines.append(line)

    return bad_lines


def validate_stderr(lines: list[str]) -> list[str]:
    return [line for line in lines if STRUCTURED_RX.match(line)]


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate strict inference stdout/stderr log compliance.")
    parser.add_argument("--stdout", default="stdout_check.log", help="Path to stdout log file")
    parser.add_argument("--stderr", default="stderr_check.log", help="Path to stderr log file")
    args = parser.parse_args()

    stdout_path = Path(args.stdout)
    stderr_path = Path(args.stderr)

    if not stdout_path.exists():
        print(f"[FAIL] Missing stdout log: {stdout_path}")
        return 1
    if not stderr_path.exists():
        print(f"[FAIL] Missing stderr log: {stderr_path}")
        return 1

    stdout_lines = read_lines(stdout_path)
    stderr_lines = read_lines(stderr_path)

    bad_stdout = validate_stdout(stdout_lines)
    bad_stderr = validate_stderr(stderr_lines)

    print(f"stdout_total={len(stdout_lines)}")
    print(f"stdout_bad={len(bad_stdout)}")
    print(f"stderr_total={len(stderr_lines)}")
    print(f"stderr_structured={len(bad_stderr)}")

    if bad_stdout:
        print("bad_stdout_lines:")
        for line in bad_stdout[:20]:
            print(line)
    if bad_stderr:
        print("bad_stderr_structured_lines:")
        for line in bad_stderr[:20]:
            print(line)

    if bad_stdout or bad_stderr:
        print("[FAIL] Inference log compliance check failed.")
        return 1

    print("[PASS] Inference log compliance check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
