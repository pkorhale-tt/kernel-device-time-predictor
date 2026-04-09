"""
step1_extract_ops.py
--------------------
Parses a TT-Metal test file and emits a YAML of raw matmul ops.

Usage:
    python step1_extract_ops.py vit_full_hw.py --model vit-base --out ops_raw.yaml

Output YAML schema (one entry per op):
    - op_id: 1
      source_test: test_vit_attention
      description: "Attention Q proj ..."
      batch: 8
      M: 224
      K: 768
      N: 768
      dtype_a: BFLOAT16
      dtype_b: BFLOAT16
      layout_a: INTERLEAVED
      layout_b: INTERLEAVED
      buffer_a: DRAM
      buffer_b: DRAM
      buffer_out: DRAM
      layout_out: INTERLEAVED
"""

import ast
import re
import sys
import copy
import yaml
import argparse
from pathlib import Path

# ── Model configs ──────────────────────────────────────────────────────────────

MODEL_CONFIGS = {
    "vit-base": dict(
        hidden_size=768, intermediate_size=3072,
        num_attention_heads=12, num_hidden_layers=12,
        seq_len_padded=224,   # ceil(197/32)*32
        head_dim=64,
    ),
    "vit-large": dict(
        hidden_size=1024, intermediate_size=4096,
        num_attention_heads=16, num_hidden_layers=24,
        seq_len_padded=224,
        head_dim=64,
    ),
    "bert-base": dict(
        hidden_size=768, intermediate_size=3072,
        num_attention_heads=12, num_hidden_layers=12,
        seq_len_padded=128,
        head_dim=64,
    ),
}

# ── Parametrize extractor ──────────────────────────────────────────────────────

def extract_parametrize(func_node):
    params = {}
    for dec in func_node.decorator_list:
        if not isinstance(dec, ast.Call):
            continue
        if not (isinstance(dec.func, ast.Attribute) and dec.func.attr == "parametrize"):
            continue
        if len(dec.args) < 2:
            continue
        key_node, val_node = dec.args[0], dec.args[1]
        if not isinstance(key_node, ast.Constant):
            continue
        key = key_node.value
        if isinstance(val_node, ast.List):
            vals = [e.value for e in val_node.elts if isinstance(e, ast.Constant)]
        elif isinstance(val_node, ast.Constant):
            vals = [val_node.value]
        else:
            vals = []
        params[key] = vals[0] if vals else None
    return params

# ── Op builder helpers ─────────────────────────────────────────────────────────

_id_counter = [0]

def _op(desc, source, batch, M, K, N,
        batch_b=1, buf_out="DRAM", layout_out="INTERLEAVED"):
    _id_counter[0] += 1
    return dict(
        op_id=_id_counter[0],
        source_test=source,
        description=desc,
        batch=batch,
        M=M, K=K, N=N,
        batch_b=batch_b,
        dtype_a="BFLOAT16", dtype_b="BFLOAT16",
        layout_a="INTERLEAVED", buffer_a="DRAM",
        layout_b="INTERLEAVED", buffer_b="DRAM",
        layout_out=layout_out, buffer_out=buf_out,
    )

# ── ViT shape inference ────────────────────────────────────────────────────────

def infer_vit_ops(fname, cfg, batch):
    S  = cfg["seq_len_padded"]
    H  = cfg["hidden_size"]
    I  = cfg["intermediate_size"]
    Nh = cfg["num_attention_heads"]
    Dh = cfg["head_dim"]
    L  = cfg["num_hidden_layers"]
    ops = []

    def mk(desc, M, K, N, ba=None, bb=1):
        return _op(desc, fname, batch if ba is None else ba, M, K, N, batch_b=bb)

    if "patch_embedding" in fname:
        ops.append(mk(f"PatchEmbed proj patches=196 C={H}", 196, H, H))

    elif "attention" in fname:
        ops.append(mk(f"Attn Q proj  S={S} H={H}", S, H, H))
        ops.append(mk(f"Attn K proj  S={S} H={H}", S, H, H))
        ops.append(mk(f"Attn V proj  S={S} H={H}", S, H, H))
        ops.append(mk(f"Attn QK^T    Nh={Nh} S={S} Dh={Dh}",
                       S, Dh, S, ba=batch*Nh, bb=batch*Nh))
        ops.append(mk(f"Attn AV      Nh={Nh} S={S} Dh={Dh}",
                       S, S, Dh, ba=batch*Nh, bb=batch*Nh))
        ops.append(mk(f"Attn out proj S={S} H={H}", S, H, H))

    elif "intermediate" in fname:
        ops.append(mk(f"FFN up-proj  S={S} H={H} I={I}", S, H, I))

    elif fname.endswith(("output", "vit_output")):
        ops.append(mk(f"FFN down-proj S={S} I={I} H={H}", S, I, H))

    elif "layer" in fname and "encoder" not in fname:
        ops += infer_vit_ops(fname.replace("layer", "attention"), cfg, batch)
        ops += infer_vit_ops(fname.replace("layer", "intermediate"), cfg, batch)
        ops += infer_vit_ops(fname.replace("layer", "vit_output"), cfg, batch)

    elif "encoder" in fname:
        # Save counter state before building template layer, then re-stamp IDs per layer
        _id_counter[0] = 0  # reset temporarily; analyze() will renumber all at the end
        per_layer = (
            infer_vit_ops("test_vit_attention",    cfg, batch) +
            infer_vit_ops("test_vit_intermediate", cfg, batch) +
            infer_vit_ops("test_vit_output",       cfg, batch)
        )
        _id_counter[0] = 0  # discard template IDs; analyze() renumbers everything
        for i in range(L):
            for op in per_layer:
                o = copy.deepcopy(op)
                o["source_test"] = fname
                o["description"] = f"[L{i:02d}] " + op["description"]
                ops.append(o)

    elif fname.endswith("test_vit"):
        ops += infer_vit_ops("test_vit_patch_embeddings", cfg, batch)
        ops += infer_vit_ops("test_vit_encoder",          cfg, batch)
        ops.append(mk(f"Classifier head H={H}→1000", 1, H, 1000))

    return ops

# ── File analyzer ──────────────────────────────────────────────────────────────

def sanitize_source(src):
    """Strip trailing junk after ) on decorator lines (e.g. '[3])can ')"""
    lines = []
    for line in src.splitlines():
        m = re.match(r'^(\s*@pytest\.mark\.parametrize\([^)]+\))\s*\w+.*$', line)
        lines.append(m.group(1) if m else line)
    return "\n".join(lines)

def analyze(path: Path, model_key: str):
    source = sanitize_source(path.read_text())
    tree   = ast.parse(source)
    cfg    = MODEL_CONFIGS.get(model_key, MODEL_CONFIGS["vit-base"])
    all_ops = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or not node.name.startswith("test_"):
            continue
        params = extract_parametrize(node)
        batch  = params.get("batch_size", 8) or 8
        ops    = infer_vit_ops(node.name, cfg, batch)
        if not ops:
            ops = [dict(
                op_id=None,
                source_test=node.name,
                description=f"[TODO] {node.name}",
                batch=batch, M=None, K=None, N=None,
                batch_b=1,
                dtype_a="BFLOAT16", dtype_b="BFLOAT16",
                layout_a="INTERLEAVED", buffer_a="DRAM",
                layout_b="INTERLEAVED", buffer_b="DRAM",
                layout_out="INTERLEAVED", buffer_out="DRAM",
            )]
        all_ops.extend(ops)

    # Renumber all ops sequentially — avoids any double-increment from recursive calls
    for i, op in enumerate(all_ops, start=1):
        op["op_id"] = i
    return all_ops

# ── YAML writer ───────────────────────────────────────────────────────────────

class _Dumper(yaml.Dumper):
    pass
_Dumper.add_representer(
    bool,
    lambda d, v: d.represent_scalar("tag:yaml.org,2002:bool", "true" if v else "false")
)

def write_yaml(records, path):
    with open(path, "w") as f:
        yaml.dump(records, f, Dumper=_Dumper,
                  default_flow_style=False, sort_keys=False, allow_unicode=True)

# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Step 1 — extract matmul ops from test file")
    p.add_argument("test_file", help="Path to test_*.py")
    p.add_argument("--model", default="vit-base", choices=list(MODEL_CONFIGS))
    p.add_argument("--out",   default="ops_raw.yaml")
    args = p.parse_args()

    ops = analyze(Path(args.test_file), args.model)
    write_yaml(ops, args.out)

    valid = [o for o in ops if o.get("M")]
    print(f"Extracted {len(valid)} ops  →  {args.out}")
    print(f"\n{'ID':>4}  {'batch':>5}  {'M':>5}  {'K':>5}  {'N':>5}  Description")
    print("─" * 72)
    for o in valid:
        print(f"{o['op_id']:>4}  {o['batch']:>5}  {o['M']:>5}  {o['K']:>5}  {o['N']:>5}  {o['description'][:40]}")

if __name__ == "__main__":
    main()
