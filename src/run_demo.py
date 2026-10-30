#!/usr/bin/env python3
"""Run the offline demo: score the synthetic results file and print a report.

    python src/run_demo.py                 # report to stdout
    python src/run_demo.py --out report.md # and write it to a file

The demo data is fully synthetic (invented endpoints, invented token
counts). One record is consistent with its label, the other contradicts
its label on two independent signals and earns a MISMATCH.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from probe import load_results, render_report  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "examples" / "demo_results.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    data = load_results(args.input)
    report = render_report(data)
    print(report, end="")
    if args.out:
        Path(args.out).write_text(report, encoding="utf-8")
    # The demo is also a smoke test: the synthetic mismatch must be caught.
    if "endpoint-beta" in report and "MISMATCH" not in report:
        print("demo check failed: synthetic mismatch was not detected", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
