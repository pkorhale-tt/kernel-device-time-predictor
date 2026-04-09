"""
step4_report.py
---------------
Reads ops_timed.yaml (from step3) and produces:
  1. A human-readable console summary table
  2. A CSV (ops_report.csv) for further analysis / comparison vs actuals

Usage:
    python step4_report.py ops_timed.yaml [--csv ops_report.csv] [--actual actuals.csv]

If --actual is provided, it tries to match rows by (M, K, N, batch) and
compute prediction error against DEVICE KERNEL DURATION [ns].
"""

import sys
import yaml
import argparse
import csv
from pathlib import Path

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)

def cfg_short(cfg_type):
    if "1D"  in str(cfg_type): return "1D"
    if "2D"  in str(cfg_type): return "2D"
    if "Reuse" in str(cfg_type): return "REUSE"
    return str(cfg_type)[:8]

def fmt_ns(ns):
    if ns is None: return "N/A"
    if ns >= 1_000_000: return f"{ns/1e6:.2f}ms"
    if ns >= 1_000:     return f"{ns/1e3:.1f}µs"
    return f"{ns}ns"

# ── Console report ─────────────────────────────────────────────────────────────

def print_report(ops):
    valid = [o for o in ops if o.get("predicted_ns")]

    print("\n" + "═" * 108)
    print(f"  VIT KERNEL TIME PREDICTION REPORT  —  {len(valid)} ops")
    print("═" * 108)
    print(f"{'ID':>4}  {'batch':>5}  {'M':>5}  {'K':>5}  {'N':>5}  "
          f"{'cfg':>5}  {'cores':>5}  {'pcM':>3}  {'pcN':>3}  "
          f"{'pred_ns':>12}  {'Bot':>3}  {'BW':>4}  description")
    print("─" * 108)

    total_ns = 0
    for o in ops:
        ns  = o.get("predicted_ns")
        bot = (o.get("bottleneck") or "")[:3]
        bw  = o.get("eff_bw_GBs", "")
        total_ns += ns or 0
        print(
            f"{str(o.get('op_id') or ''):>4}  "
            f"{str(o.get('batch') or 1):>5}  "
            f"{str(o.get('M') or '?'):>5}  "
            f"{str(o.get('K') or '?'):>5}  "
            f"{str(o.get('N') or '?'):>5}  "
            f"{cfg_short(o.get('config_type','')):>5}  "
            f"{str(o.get('predicted_cores') or ''):>5}  "
            f"{str(o.get('per_core_M') or ''):>3}  "
            f"{str(o.get('per_core_N') or ''):>3}  "
            f"{str(ns or 'N/A'):>12}  "
            f"{bot:>3}  "
            f"{str(bw):>4}  "
            f"{o.get('description','')[:36]}"
        )

    print("═" * 108)
    print(f"  Total predicted time: {total_ns:,} ns  ({total_ns/1e6:.3f} ms)")
    print()

    # Summary by config type
    from collections import defaultdict
    by_cfg = defaultdict(list)
    for o in valid:
        by_cfg[cfg_short(o.get("config_type", "?"))].append(o.get("predicted_ns", 0))
    print("  Time by config type:")
    for k, v in sorted(by_cfg.items(), key=lambda x: -sum(x[1])):
        s = sum(v)
        print(f"    {k:<8}  {len(v):>3} ops   {s:>14,} ns  ({s/1e6:.2f} ms)")

    # Bottleneck split
    compute = sum(1 for o in valid if o.get("bottleneck") == "COMPUTE")
    bw_     = sum(1 for o in valid if o.get("bottleneck") == "BANDWIDTH")
    print(f"\n  Bottleneck: {compute} compute-bound, {bw_} bandwidth-bound")
    print()

# ── CSV export ────────────────────────────────────────────────────────────────

CSV_FIELDS = [
    "op_id", "source_test", "description",
    "batch", "M", "K", "N",
    "config_type", "predicted_cores", "per_core_M", "per_core_N",
    "in0_block_w", "mcast_in0", "fuse_batch",
    "predicted_ns", "compute_cyc_per_k", "reader_cyc_per_k",
    "bottleneck", "total_bytes_per_core", "eff_bw_GBs",
]

def write_csv(ops, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for o in ops:
            w.writerow({k: o.get(k, "") for k in CSV_FIELDS})
    print(f"CSV written: {path}")

# ── Accuracy check vs actual CSV ──────────────────────────────────────────────

def compare_actuals(ops, actual_csv):
    if not HAS_PANDAS:
        print("[WARN] pandas not available — skipping accuracy check")
        return
    import pandas as pd
    import re

    def parse_dim(v):
        if pd.isna(v): return 1
        m = re.match(r'^(\d+)', str(v).strip())
        return int(m.group(1)) if m else 1

    df = pd.read_csv(actual_csv)
    mask = df["OP CODE"].str.contains("matmul|Matmul", case=False, na=False)
    df = df[mask].copy()

    df["_M"] = df["INPUT_0_Y_PAD[LOGICAL]"].apply(parse_dim)
    df["_K"] = df["INPUT_0_X_PAD[LOGICAL]"].apply(parse_dim)
    df["_N"] = df["INPUT_1_X_PAD[LOGICAL]"].apply(parse_dim)
    df["_B"] = (df["INPUT_0_W_PAD[LOGICAL]"].apply(parse_dim) *
                df["INPUT_0_Z_PAD[LOGICAL]"].apply(parse_dim))

    errors = []
    print(f"\n  Accuracy vs {actual_csv}:")
    print(f"  {'M':>5}  {'K':>5}  {'N':>5}  {'batch':>5}  "
          f"{'actual_ns':>12}  {'pred_ns':>12}  {'err%':>7}")
    print("  " + "─" * 65)

    for op in ops:
        M, K, N, B = op.get("M"), op.get("K"), op.get("N"), op.get("batch", 1)
        pred = op.get("predicted_ns")
        if not all([M, K, N, pred]):
            continue
        match = df[(df["_M"] == M) & (df["_K"] == K) &
                   (df["_N"] == N) & (df["_B"] == B)]
        if match.empty:
            continue
        actual = float(match["DEVICE KERNEL DURATION [ns]"].iloc[0])
        err = (pred - actual) / actual * 100
        errors.append(abs(err))
        print(f"  {M:>5}  {K:>5}  {N:>5}  {B:>5}  "
              f"{actual:>12,.0f}  {pred:>12,}  {err:>+7.1f}%")

    if errors:
        import statistics
        print(f"\n  Mean error : {sum(errors)/len(errors):.1f}%")
        print(f"  Median     : {statistics.median(errors):.1f}%")
        print(f"  Within ±10%: {sum(1 for e in errors if e <= 10)} / {len(errors)}")
        print(f"  Within ±20%: {sum(1 for e in errors if e <= 20)} / {len(errors)}")

# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Step 4 — report kernel time predictions")
    p.add_argument("timed_yaml", help="Input: ops_timed.yaml from step3")
    p.add_argument("--csv",    default="ops_report.csv", help="Output CSV path")
    p.add_argument("--actual", default=None,
                   help="Optional: actual perf CSV to compare predictions against")
    args = p.parse_args()

    ops = load_yaml(args.timed_yaml)
    print_report(ops)
    write_csv(ops, args.csv)

    if args.actual:
        compare_actuals(ops, args.actual)

if __name__ == "__main__":
    main()
