#!/usr/bin/env python3
"""
run_pipeline.py
---------------
Master runner: chains all 4 steps end-to-end.

Usage:
    python run_pipeline.py vit_full_hw.py [OPTIONS]

Options:
    --model         vit-base | vit-large | bert-base   (default: vit-base)
    --fidelity      LoFi | HiFi2 | HiFi3 | HiFi4      (default: HiFi2)
    --actual        path to perf CSV (optional accuracy check)
    --workdir       where to write intermediate YAMLs    (default: ./pipeline_out)
    --kernel-config path to dir with matmul_config_wrapper.py
    --keep          keep intermediate YAML files          (default: yes)

Intermediate files (in --workdir):
    ops_raw.yaml          ← step1: shapes extracted from test file
    ops_configured.yaml   ← step2: + config type, per_core_M/N, cores
    ops_timed.yaml        ← step3: + predicted_ns, bottleneck, bw
    ops_report.csv        ← step4: flat CSV for spreadsheet analysis

Example:
    python run_pipeline.py vit_full_hw.py \\
        --model vit-base \\
        --fidelity HiFi2 \\
        --actual /path/to/perf_data.csv \\
        --kernel-config /proj_sw/user_dev/pkorhale/tt-metal/kernel_config
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

HERE = Path(__file__).parent

STEPS = [
    ("step1_extract_ops.py",     "Step 1: Extract ops from test file"),
    ("step2_predict_config.py",  "Step 2: Predict program config + cores"),
    ("step3_predict_time.py",    "Step 3: Predict kernel time"),
    ("step4_report.py",          "Step 4: Generate report"),
]

BANNER = "═" * 70

def run_step(script, args_list, label):
    print(f"\n{BANNER}")
    print(f"  {label}")
    print(BANNER)
    cmd = [sys.executable, str(HERE / script)] + args_list
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\n[ERROR] {label} failed (exit {result.returncode})")
        sys.exit(result.returncode)

def main():
    p = argparse.ArgumentParser(
        description="End-to-end VIT matmul kernel time predictor",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("test_file",          help="Path to TT-Metal test file (test_*.py)")
    p.add_argument("--model",            default="vit-base",
                   choices=["vit-base", "vit-large", "bert-base"])
    p.add_argument("--fidelity",         default="HiFi2",
                   choices=["LoFi", "HiFi2", "HiFi3", "HiFi4"])
    p.add_argument("--actual",           default=None,
                   help="Perf CSV with actual DEVICE KERNEL DURATION [ns]")
    p.add_argument("--workdir",          default="pipeline_out")
    p.add_argument("--kernel-config",    default=None,
                   help="Path to dir containing matmul_config_wrapper.py")
    p.add_argument("--csv-out",          default=None,
                   help="Final CSV output path (default: workdir/ops_report.csv)")
    args = p.parse_args()

    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    raw_yaml    = workdir / "ops_raw.yaml"
    cfg_yaml    = workdir / "ops_configured.yaml"
    timed_yaml  = workdir / "ops_timed.yaml"
    report_csv  = Path(args.csv_out) if args.csv_out else workdir / "ops_report.csv"

    print(f"\n{BANNER}")
    print("  VIT MATMUL KERNEL TIME PREDICTION PIPELINE")
    print(BANNER)
    print(f"  Test file  : {args.test_file}")
    print(f"  Model      : {args.model}")
    print(f"  Fidelity   : {args.fidelity}")
    print(f"  Work dir   : {workdir}")
    if args.actual:
        print(f"  Actual CSV : {args.actual}")
    if args.kernel_config:
        print(f"  Config dir : {args.kernel_config}")

    # ── Step 1 ────────────────────────────────────────────────────────────────
    run_step("step1_extract_ops.py",
             [args.test_file, "--model", args.model, "--out", str(raw_yaml)],
             STEPS[0][1])

    # ── Step 2 ────────────────────────────────────────────────────────────────
    s2_args = [str(raw_yaml), "--out", str(cfg_yaml)]
    if args.kernel_config:
        s2_args += ["--kernel-config-dir", args.kernel_config]
    run_step("step2_predict_config.py", s2_args, STEPS[1][1])

    # ── Step 3 ────────────────────────────────────────────────────────────────
    run_step("step3_predict_time.py",
             [str(cfg_yaml), "--out", str(timed_yaml), "--fidelity", args.fidelity],
             STEPS[2][1])

    # ── Step 4 ────────────────────────────────────────────────────────────────
    s4_args = [str(timed_yaml), "--csv", str(report_csv)]
    if args.actual:
        s4_args += ["--actual", args.actual]
    run_step("step4_report.py", s4_args, STEPS[3][1])

    print(f"\n{BANNER}")
    print("  PIPELINE COMPLETE")
    print(BANNER)
    print(f"  Intermediate YAMLs : {workdir}/")
    print(f"  Final CSV          : {report_csv}")
    print()

if __name__ == "__main__":
    main()
