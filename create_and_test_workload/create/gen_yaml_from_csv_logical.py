#!/usr/bin/env python3
"""
Pure CSV → YAML Workload Generator
-----------------------------------
Extracts EXACT inputs from CSV without any auto-adjustment.
Uses LOGICAL dimensions (inside brackets) not padded dimensions.

e.g. '224[197]' → 197  (logical, what the user passed)
                        (let TT-Metal handle padding internally)

Usage:
    python gen_yaml_from_csv.py input.csv output.yaml
"""

import sys
import re
import pandas as pd
import yaml

TILE = 32

# ─────────────────────────────────────────────────────────────────
# Parsing helpers
# ─────────────────────────────────────────────────────────────────

def parse_dim_logical(val) -> int:
    """
    Parse logical dimension from padded string.

    Examples:
        '224[197]'  → 197   (logical value inside brackets)
        '1024[1024]'→ 1024  (same when no difference)
        '8[8]'      → 8
        '32'        → 32    (no brackets → use as-is)
    """
    if pd.isna(val):
        return 1
    s = str(val).strip()
    # Try to get value inside brackets first: '224[197]' → 197
    m = re.search(r'\[(\d+)\]', s)
    if m:
        return int(m.group(1))
    # No brackets → just parse the number directly
    m = re.match(r'^(\d+)', s)
    return int(m.group(1)) if m else 1


def parse_memory(mem_str: str) -> tuple:
    """'DEV_1_DRAM_INTERLEAVED' → ('DRAM', 'INTERLEAVED')"""
    if pd.isna(mem_str):
        return "DRAM", "INTERLEAVED"

    parts = str(mem_str).strip().split("_")
    rest = parts[2:]  # Skip 'DEV' and device ID

    buffer_type = rest[0] if rest else "DRAM"
    layout = "_".join(rest[1:]) if len(rest) > 1 else "INTERLEAVED"

    buffer_type = "L1" if buffer_type == "L1" else "DRAM"
    valid_layouts = {"INTERLEAVED", "HEIGHT_SHARDED", "WIDTH_SHARDED", "BLOCK_SHARDED"}
    layout = layout if layout in valid_layouts else "INTERLEAVED"

    return buffer_type, layout


def parse_attributes(attrs_str: str) -> dict:
    """Extract compute flags from ATTRIBUTES column."""
    result = {
        "transpose_a":      False,
        "transpose_b":      False,
        "fp32_dest_acc_en": False,
        "packer_l1_acc":    True,
        "math_fidelity":    "HiFi2",
        "math_approx_mode": False,
        "user_run_batched": False,
    }

    if pd.isna(attrs_str):
        return result

    s = str(attrs_str)

    for key in ("transpose_a", "transpose_b"):
        m = re.search(rf"'{key}': '(true|false)'", s)
        if m:
            result[key] = m.group(1).lower() == "true"

    m = re.search(r"fp32_dest_acc_en=(\d)", s)
    if m:
        result["fp32_dest_acc_en"] = m.group(1) == "1"

    m = re.search(r"packer_l1_acc=(\d)", s)
    if m:
        result["packer_l1_acc"] = m.group(1) == "1"

    m = re.search(r"math_fidelity=(\w+)", s)
    if m:
        result["math_fidelity"] = m.group(1)

    m = re.search(r"math_approx_mode=(\d)", s)
    if m:
        result["math_approx_mode"] = m.group(1) == "1"

    m = re.search(r"'user_run_batched': '(true|false)'", s)
    if m:
        result["user_run_batched"] = m.group(1).lower() == "true"

    return result


# ─────────────────────────────────────────────────────────────────
# Extract inputs using LOGICAL dimensions
# ─────────────────────────────────────────────────────────────────

def extract_inputs(row) -> dict:
    """
    Extract inputs from CSV row using LOGICAL dimensions.

    LOGICAL = the value inside brackets e.g. '224[197]' → 197
    This is what the user originally passed to ttnn.matmul().
    TT-Metal will handle padding internally when we run the workload.
    """
    attrs = parse_attributes(row.get("ATTRIBUTES", ""))

    # ── Tensor A ─────────────────────────────────────────────────
    a_w = parse_dim_logical(row.get("INPUT_0_W_PAD[LOGICAL]"))
    a_z = parse_dim_logical(row.get("INPUT_0_Z_PAD[LOGICAL]"))
    a_y = parse_dim_logical(row.get("INPUT_0_Y_PAD[LOGICAL]"))
    a_x = parse_dim_logical(row.get("INPUT_0_X_PAD[LOGICAL]"))
    a_buf, a_layout = parse_memory(row.get("INPUT_0_MEMORY"))

    tensor_a = {
        "batch":          a_w * a_z,
        "M":              a_y,          # logical M (not padded)
        "K":              a_x,          # logical K
        "dtype":          str(row.get("INPUT_0_DATATYPE", "BFLOAT16")).strip(),
        "memory_layout":  a_layout,
        "buffer_type":    a_buf,
        "transpose":      attrs["transpose_a"],
    }

    # ── Tensor B ─────────────────────────────────────────────────
    b_w = parse_dim_logical(row.get("INPUT_1_W_PAD[LOGICAL]"))
    b_z = parse_dim_logical(row.get("INPUT_1_Z_PAD[LOGICAL]"))
    b_y = parse_dim_logical(row.get("INPUT_1_Y_PAD[LOGICAL]"))
    b_x = parse_dim_logical(row.get("INPUT_1_X_PAD[LOGICAL]"))
    b_buf, b_layout = parse_memory(row.get("INPUT_1_MEMORY"))

    tensor_b = {
        "batch":          b_w * b_z,
        "K":              b_y,          # logical K (must match A's K)
        "N":              b_x,          # logical N
        "dtype":          str(row.get("INPUT_1_DATATYPE", "BFLOAT16")).strip(),
        "memory_layout":  b_layout,
        "buffer_type":    b_buf,
        "transpose":      attrs["transpose_b"],
    }

    # ── Output memory ─────────────────────────────────────────────
    o_buf, o_layout = parse_memory(row.get("OUTPUT_0_MEMORY"))
    o_dtype = str(row.get("OUTPUT_0_DATATYPE",
                           tensor_a["dtype"])).strip()

    output = {
        "dtype":          o_dtype,
        "memory_layout":  o_layout,
        "buffer_type":    o_buf,
    }

    # ── Compute config ────────────────────────────────────────────
    compute = {
        "math_fidelity":    attrs["math_fidelity"],
        "fp32_dest_acc_en": attrs["fp32_dest_acc_en"],
        "packer_l1_acc":    attrs["packer_l1_acc"],
        "math_approx_mode": attrs["math_approx_mode"],
    }

    flags = {
        "user_run_batched": attrs["user_run_batched"],
    }

    return {
        "tensor_a": tensor_a,
        "tensor_b": tensor_b,
        "output":   output,
        "compute":  compute,
        "flags":    flags,
    }


# ─────────────────────────────────────────────────────────────────
# Row → Op (no program_config, let TT-Metal decide)
# ─────────────────────────────────────────────────────────────────

def row_to_op(row, op_id: int) -> dict:
    inputs = extract_inputs(row)
    a = inputs["tensor_a"]
    b = inputs["tensor_b"]

    return {
        "op_id":       op_id,
        "description": (f"Matmul M={a['M']} K={a['K']} N={b['N']} "
                        f"batch_a={a['batch']} batch_b={b['batch']}"),
        "tensor_a":    a,
        "tensor_b":    b,
        "output":      inputs["output"],
        "compute":     inputs["compute"],
        "flags":       inputs["flags"],
        # NO program_config — TT-Metal auto-selects
    }


# ─────────────────────────────────────────────────────────────────
# Deduplication
# ─────────────────────────────────────────────────────────────────

def make_key(row) -> tuple:
    inputs = extract_inputs(row)
    a = inputs["tensor_a"]
    b = inputs["tensor_b"]
    return (
        a["batch"], b["batch"],
        a["M"], a["K"], b["N"],
        a["memory_layout"], b["memory_layout"],
        inputs["compute"]["math_fidelity"],
    )


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) != 3:
        print("Usage: python gen_yaml_from_csv.py <input.csv> <output.yaml>")
        sys.exit(1)

    in_path  = sys.argv[1]
    out_path = sys.argv[2]

    print(f"\n{'═'*60}")
    print(f"  CSV → YAML Generator  (logical dims)")
    print(f"{'═'*60}")
    print(f"Reading: {in_path}")

    df = pd.read_csv(in_path)
    print(f"Total rows: {len(df)}")

    mat = df[df["OP CODE"].str.contains(
        "matmul|Matmul", case=False, na=False)].copy()
    print(f"Matmul rows: {len(mat)}")

    seen = set()
    ops  = []
    op_id = 1

    for _, row in mat.iterrows():
        key = make_key(row)
        if key not in seen:
            seen.add(key)
            try:
                op = row_to_op(row, op_id)
                ops.append(op)
                print(f"  ✓ [{op_id:2d}] {op['description']}")
                op_id += 1
            except Exception as e:
                print(f"  ✗ Skipping row: {e}")

    print(f"\nUnique ops: {len(ops)}")

    workload = {
        "workload": {
            "name":        in_path,
            "total_ops":   len(ops),
            "description": "Logical dims — TT-Metal handles padding",
            "note":        "Dimensions are LOGICAL (inside brackets), "
                           "not padded. TT-Metal auto-selects program config.",
        },
        "ops": ops,
    }

    with open(out_path, "w") as f:
        yaml.dump(workload, f,
                  default_flow_style=False,
                  sort_keys=False,
                  indent=2)

    print(f"\n{'═'*60}")
    print(f"  YAML written : {out_path}")
    print(f"  Dims used    : LOGICAL (inside brackets)")
    print(f"  Program config: NOT included — TT-Metal decides")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()
