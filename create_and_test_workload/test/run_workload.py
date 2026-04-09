#!/usr/bin/env python3
"""
YAML Workload Runner - Auto Mode
---------------------------------
Runs matmul with ONLY tensor specs from YAML.
TT-Metal auto-selects program config.

Usage:
    python run_workload.py workload.yaml
"""

import sys
import argparse
import time
import yaml
import torch

import ttnn

TILE = 32

DTYPE_MAP = {
    "BFLOAT16": ttnn.bfloat16,
    "BFLOAT8_B": ttnn.bfloat8_b,
    "BFLOAT4_B": ttnn.bfloat4_b,
    "FLOAT32":  ttnn.float32,
}

MATH_FIDELITY_MAP = {
    "LoFi":  ttnn.MathFidelity.LoFi,
    "HiFi2": ttnn.MathFidelity.HiFi2,
    "HiFi3": ttnn.MathFidelity.HiFi3,
    "HiFi4": ttnn.MathFidelity.HiFi4,
}

# ══════════════════════════════════════════════════════════════════
# Memory Config Builder
# ══════════════════════════════════════════════════════════════════
def make_mem_config(layout: str, buffer_type: str) -> ttnn.MemoryConfig:
    """Build MemoryConfig from layout and buffer type strings."""
    buf = ttnn.BufferType.L1 if buffer_type == "L1" else ttnn.BufferType.DRAM
    
    layout_map = {
        "INTERLEAVED":    ttnn.TensorMemoryLayout.INTERLEAVED,
        "HEIGHT_SHARDED": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        "WIDTH_SHARDED":  ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        "BLOCK_SHARDED":  ttnn.TensorMemoryLayout.BLOCK_SHARDED,
    }
    
    mem_layout = layout_map.get(layout, ttnn.TensorMemoryLayout.INTERLEAVED)
    return ttnn.MemoryConfig(mem_layout, buf)


# ══════════════════════════════════════════════════════════════════
# Compute Kernel Config Builder
# ══════════════════════════════════════════════════════════════════
def make_compute_kernel_config(compute: dict):
    """Build WormholeComputeKernelConfig from YAML dict."""
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity    = MATH_FIDELITY_MAP.get(compute["math_fidelity"], ttnn.MathFidelity.HiFi2),
        math_approx_mode = compute.get("math_approx_mode", False),
        fp32_dest_acc_en = compute.get("fp32_dest_acc_en", False),
        packer_l1_acc    = compute.get("packer_l1_acc", True),
    )


# ══════════════════════════════════════════════════════════════════
# Tensor Builder
# ══════════════════════════════════════════════════════════════════
def make_tensor(device, shape_4d: tuple, dtype_str: str, mem_layout: str, 
                buf_type: str, verbose: bool = False) -> ttnn.Tensor:
    """Create a random tensor on device with specified config."""
    dtype = DTYPE_MAP.get(dtype_str, ttnn.bfloat16)
    mem_cfg = make_mem_config(mem_layout, buf_type)
    
    if verbose:
        print(f"    Creating tensor {shape_4d} dtype={dtype_str} mem={mem_layout}/{buf_type}")
    
    # Create on CPU
    torch_t = torch.randn(shape_4d, dtype=torch.bfloat16)
    tt_t = ttnn.from_torch(torch_t, dtype=dtype, layout=ttnn.TILE_LAYOUT)
    
    # Move to device
    tt_t = ttnn.to_device(tt_t, device, memory_config=mem_cfg)
    return tt_t


# ══════════════════════════════════════════════════════════════════
# Run Single Operation - AUTO MODE
# ══════════════════════════════════════════════════════════════════
def run_op(device, op: dict, warmup: int, runs: int, verbose: bool = False) -> dict:
    """
    Run matmul with AUTO program config selection.
    Only uses tensor specs from YAML - no manual config.
    """
    op_id = op["op_id"]
    a_cfg = op["tensor_a"]
    b_cfg = op["tensor_b"]
    o_cfg = op["output"]
    comp = op["compute"]
    
    # ── Build 4D shapes ───────────────────────────────────────────
    batch_a = a_cfg["batch"]
    M = a_cfg["M"]
    K = a_cfg["K"]
    
    batch_b = b_cfg["batch"]
    N = b_cfg["N"]
    
    shape_a = (1, batch_a, M, K)
    shape_b = (1, batch_b, K, N)
    
    # ── Build configs ─────────────────────────────────────────────
    compute_kernel_cfg = make_compute_kernel_config(comp)
    output_mem_cfg = make_mem_config(o_cfg["memory_layout"], o_cfg["buffer_type"])
    output_dtype = DTYPE_MAP.get(o_cfg["dtype"], ttnn.bfloat16)
    
    # ── Print operation info ──────────────────────────────────────
    print(f"\n{'─'*70}")
    print(f"  Op {op_id}: {op['description']}")
    print(f"{'─'*70}")
    print(f"  Shape A: {shape_a}")
    print(f"  Shape B: {shape_b}")
    print(f"  Compute: {comp['math_fidelity']}, fp32_acc={comp['fp32_dest_acc_en']}")
    print(f"  Mode:    AUTO (TT-Metal selects program config)")
    
    try:
        # ── Create tensors ────────────────────────────────────────
        print(f"  Creating tensors...")
        tt_a = make_tensor(device, shape_a, a_cfg["dtype"], 
                          a_cfg["memory_layout"], a_cfg["buffer_type"], verbose)
        tt_b = make_tensor(device, shape_b, b_cfg["dtype"], 
                          b_cfg["memory_layout"], b_cfg["buffer_type"], verbose)
        
        # ── Warmup ────────────────────────────────────────────────
        print(f"  Warming up ({warmup} runs)...")
        for i in range(warmup):
            # NO program_config - let TT-Metal auto-select
            out = ttnn.matmul(
                tt_a, tt_b,
                memory_config=output_mem_cfg,
                dtype=output_dtype,
                compute_kernel_config=compute_kernel_cfg,
            )
            ttnn.deallocate(out)
        
        # ── Timed runs ────────────────────────────────────────────
        print(f"  Running ({runs} iterations)...")
        durations = []
        
        for i in range(runs):
            ttnn.synchronize_device(device)
            t0 = time.perf_counter()
            
            # NO program_config - AUTO mode
            out = ttnn.matmul(
                tt_a, tt_b,
                memory_config=output_mem_cfg,
                dtype=output_dtype,
                compute_kernel_config=compute_kernel_cfg,
            )
            
            ttnn.synchronize_device(device)
            t1 = time.perf_counter()
            
            duration_us = (t1 - t0) * 1e6
            durations.append(duration_us)
            ttnn.deallocate(out)
            
            if verbose:
                print(f"    Run {i+1}/{runs}: {duration_us:.2f} µs")
        
        # ── Cleanup ───────────────────────────────────────────────
        ttnn.deallocate(tt_a)
        ttnn.deallocate(tt_b)
        
        # ── Calculate statistics ──────────────────────────────────
        durations_sorted = sorted(durations)
        result = {
            "op_id": op_id,
            "description": op["description"],
            "shape_a": list(shape_a),
            "shape_b": list(shape_b),
            "math_fidelity": comp["math_fidelity"],
            "mode": "AUTO",
            "runs": runs,
            "min_us": round(min(durations), 2),
            "max_us": round(max(durations), 2),
            "median_us": round(durations_sorted[len(durations)//2], 2),
            "mean_us": round(sum(durations)/len(durations), 2),
            "std_us": round((sum((x - sum(durations)/len(durations))**2 for x in durations) / len(durations))**0.5, 2),
        }
        
        print(f"  ✓ SUCCESS")
        print(f"    Min:    {result['min_us']:8.2f} µs")
        print(f"    Median: {result['median_us']:8.2f} µs")
        print(f"    Mean:   {result['mean_us']:8.2f} µs")
        print(f"    Max:    {result['max_us']:8.2f} µs")
        print(f"    Std:    {result['std_us']:8.2f} µs")
        
        return result
        
    except Exception as e:
        print(f"  ✗ FAILED: {str(e)[:100]}")
        return {
            "op_id": op_id,
            "description": op["description"],
            "error": str(e),
            "status": "FAILED",
        }


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════
def parse_args():
    parser = argparse.ArgumentParser(description="Run matmul workload in AUTO mode")
    parser.add_argument("yaml_file", help="Workload YAML file")
    parser.add_argument("--op-ids", nargs="*", type=int, default=None,
                       help="Run only specific op IDs")
    parser.add_argument("--warmup", type=int, default=0,
                       help="Warmup runs per op (default: 2)")
    parser.add_argument("--runs", type=int, default=1,
                       help="Timed runs per op (default: 10)")
    parser.add_argument("--output", type=str, default=None,
                       help="Save results to YAML file")
    parser.add_argument("--verbose", action="store_true",
                       help="Print detailed execution information")
    parser.add_argument("--device-id", type=int, default=0,
                       help="Device ID to use (default: 0)")
    return parser.parse_args()


def main():
    args = parse_args()
    
    print(f"\n{'═'*70}")
    print(f"  TT-Metal Matmul Workload Runner (AUTO MODE)")
    print(f"{'═'*70}")
    print(f"Loading: {args.yaml_file}")
    
    with open(args.yaml_file) as f:
        workload = yaml.safe_load(f)
    
    ops = workload["ops"]
    workload_info = workload.get("workload", {})
    
    print(f"\nWorkload: {workload_info.get('name', 'Unknown')}")
    print(f"Total ops: {len(ops)}")
    print(f"Mode: AUTO (hardware selects program config)")
    
    if args.op_ids:
        ops = [op for op in ops if op["op_id"] in args.op_ids]
        print(f"Running ops: {args.op_ids}")
    
    print(f"Warmup: {args.warmup} | Runs: {args.runs} | Device: {args.device_id}")
    
    device = ttnn.open_device(device_id=args.device_id)
    results = []
    errors = []
    
    try:
        for op in ops:
            result = run_op(device, op, args.warmup, args.runs, args.verbose)
            if "error" in result:
                errors.append(result)
            else:
                results.append(result)
    finally:
        ttnn.close_device(device)
    
    # Print summary
    print(f"\n{'═'*70}")
    print(f"  RESULTS")
    print(f"{'═'*70}")
    print(f"Successful: {len(results)} | Failed: {len(errors)}")
    
    if results:
        print(f"\n{'ID':<4} {'Description':<40} {'Median (µs)':<12}")
        print(f"{'─'*70}")
        for r in results:
            desc = r['description'][:38]
            print(f"{r['op_id']:<4} {desc:<40} {r['median_us']:<12.2f}")
    
    if errors:
        print(f"\nFailed Operations:")
        for e in errors:
            print(f"  Op {e['op_id']}: {e['error'][:60]}")
    
    if args.output:
        with open(args.output, "w") as f:
            yaml.dump({"results": results, "errors": errors}, f)
        print(f"\nResults saved: {args.output}")
    
    print(f"{'═'*70}\n")


if __name__ == "__main__":
    main()
