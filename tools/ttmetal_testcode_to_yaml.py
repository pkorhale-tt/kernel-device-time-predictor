"""
ttmetal_testcode_to_yaml

Statically analyzes TT-Metal test files and emits a simulator YAML
for every matmul/linear op it can reconstruct from shape annotations,
parametrize decorators, and known model configs.

Usage:
    python ttmetal_to_yaml.py test_file.py [--model vit-base] [--out ops.yaml]
    python ttmetal_to_yaml.py test_file.py --manual  # drop into interactive mode
"""

import ast
import re
import sys
import argparse
import yaml
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional


# ---------------------------------------------------------------------------
# Known model configs (add more as needed)
# ---------------------------------------------------------------------------

MODEL_CONFIGS = {
    "vit-base": {
        "hidden_size": 768,
        "intermediate_size": 3072,
        "num_attention_heads": 12,
        "num_hidden_layers": 12,
        "patch_size": 16,
        "image_size": 224,
        "num_patches": 196,          # (224/16)^2
        "seq_len": 197,              # num_patches + 1 CLS token
        "seq_len_padded": 224,       # next tile multiple (tile=32 → ceil(197/32)*32=224)
        "head_dim": 64,              # hidden_size / num_heads
    },
    "vit-large": {
        "hidden_size": 1024,
        "intermediate_size": 4096,
        "num_attention_heads": 16,
        "num_hidden_layers": 24,
        "patch_size": 16,
        "image_size": 224,
        "num_patches": 196,
        "seq_len": 197,
        "seq_len_padded": 224,
        "head_dim": 64,
    },
    "bert-base": {
        "hidden_size": 768,
        "intermediate_size": 3072,
        "num_attention_heads": 12,
        "num_hidden_layers": 12,
        "seq_len": 128,
        "seq_len_padded": 128,
        "head_dim": 64,
    },
}


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class TensorSpec:
    batch: Optional[int] = None
    M: Optional[int] = None
    K: Optional[int] = None
    N: Optional[int] = None
    dtype: str = "BFLOAT16"
    memory_layout: str = "INTERLEAVED"
    buffer_type: str = "DRAM"
    transpose: bool = False


@dataclass
class ComputeSpec:
    math_fidelity: str = "HiFi2"
    fp32_dest_acc_en: bool = False
    packer_l1_acc: bool = True
    math_approx_mode: bool = False


@dataclass
class OpEntry:
    op_id: int
    description: str
    op_type: str = "matmul"           # matmul | conv | elementwise
    source_test: str = ""
    tensor_a: TensorSpec = field(default_factory=TensorSpec)
    tensor_b: TensorSpec = field(default_factory=TensorSpec)
    output: TensorSpec = field(default_factory=TensorSpec)
    compute: ComputeSpec = field(default_factory=ComputeSpec)
    flags: dict = field(default_factory=lambda: {"user_run_batched": False})


# ---------------------------------------------------------------------------
# Parametrize extractor
# ---------------------------------------------------------------------------

def extract_parametrize(func_node: ast.FunctionDef) -> dict:
    """Pull @pytest.mark.parametrize values off a function node."""
    params = {}
    for dec in func_node.decorator_list:
        if not isinstance(dec, ast.Call):
            continue
        func = dec.func
        name = ""
        if isinstance(func, ast.Attribute) and func.attr == "parametrize":
            name = "parametrize"
        if name != "parametrize":
            continue
        if len(dec.args) < 2:
            continue
        key_node, val_node = dec.args[0], dec.args[1]
        if not isinstance(key_node, ast.Constant):
            continue
        key = key_node.value
        if isinstance(val_node, ast.List):
            vals = [elt.value for elt in val_node.elts if isinstance(elt, ast.Constant)]
        elif isinstance(val_node, ast.Constant):
            vals = [val_node.value]
        else:
            vals = []
        params[key] = vals[0] if vals else None
    return params


# ---------------------------------------------------------------------------
# Shape inference rules per known ViT sub-ops
# ---------------------------------------------------------------------------

def infer_vit_ops(func_name: str, params: dict, cfg: dict, batch: int) -> list[OpEntry]:
    """
    Given a ViT test function name + parametrize values + model config,
    return a list of OpEntry objects representing the matmuls inside.
    """
    S = cfg["seq_len_padded"]   # sequence dimension (tile-aligned)
    H = cfg["hidden_size"]
    I = cfg["intermediate_size"]
    Nh = cfg["num_attention_heads"]
    Dh = cfg["head_dim"]
    L = cfg["num_hidden_layers"]
    ops = []

    def make(desc, M, K, N, batch_a=batch, batch_b=1, src=""):
        a = TensorSpec(batch=batch_a, M=M, K=K)
        b = TensorSpec(batch=batch_b, K=K, N=N)
        o = TensorSpec()
        return OpEntry(
            op_id=0,
            description=desc,
            source_test=src,
            tensor_a=a,
            tensor_b=b,
            output=o,
        )

    # --- patch embeddings: unfold + linear proj ---
    if "patch_embeddings" in func_name:
        # Each image: 196 patches, each patch = 16*16*3 = 768 → proj to 768
        ops.append(make(
            f"PatchEmbed conv-as-matmul batch={batch} patches=196 C_in=768 C_out={H}",
            M=196, K=H, N=H, src=func_name
        ))

    # --- attention: Q K V projections + scaled dot-product ---
    elif "attention" in func_name:
        # Q projection: (batch, S, H) x (H, H)
        ops.append(make(f"Attention Q proj S={S} H={H}", M=S, K=H, N=H, src=func_name))
        # K projection
        ops.append(make(f"Attention K proj S={S} H={H}", M=S, K=H, N=H, src=func_name))
        # V projection
        ops.append(make(f"Attention V proj S={S} H={H}", M=S, K=H, N=H, src=func_name))
        # QK^T: (batch*Nh, S, Dh) x (batch*Nh, Dh, S)
        ops.append(make(
            f"Attention QK^T Nh={Nh} S={S} Dh={Dh}",
            M=S, K=Dh, N=S, batch_a=batch * Nh, batch_b=batch * Nh, src=func_name
        ))
        # Attn*V: (batch*Nh, S, S) x (batch*Nh, S, Dh)
        ops.append(make(
            f"Attention AttnV Nh={Nh} S={S} Dh={Dh}",
            M=S, K=S, N=Dh, batch_a=batch * Nh, batch_b=batch * Nh, src=func_name
        ))
        # Output projection: (batch, S, H) x (H, H)
        ops.append(make(f"Attention out proj S={S} H={H}", M=S, K=H, N=H, src=func_name))

    # --- FFN intermediate (up-proj): hidden → intermediate ---
    elif "intermediate" in func_name:
        ops.append(make(
            f"FFN intermediate (up-proj) S={S} H={H} I={I}",
            M=S, K=H, N=I, src=func_name
        ))

    # --- FFN output (down-proj + residual): intermediate → hidden ---
    elif func_name.endswith("output"):
        ops.append(make(
            f"FFN output (down-proj) S={S} I={I} H={H}",
            M=S, K=I, N=H, src=func_name
        ))

    # --- full layer = attention + FFN ---
    elif "layer" in func_name and "encoder" not in func_name:
        ops += infer_vit_ops("test_vit_attention", params, cfg, batch)
        ops += infer_vit_ops("test_vit_intermediate", params, cfg, batch)
        ops += infer_vit_ops("test_vit_output", params, cfg, batch)

    # --- full encoder = L layers ---
    elif "encoder" in func_name:
        layer_ops = (
            infer_vit_ops("test_vit_attention", params, cfg, batch) +
            infer_vit_ops("test_vit_intermediate", params, cfg, batch) +
            infer_vit_ops("test_vit_output", params, cfg, batch)
        )
        for i in range(L):
            for op in layer_ops:
                import copy
                o = copy.deepcopy(op)
                o.description = f"[Layer {i}] " + o.description
                ops.append(o)

    # --- full model = embeddings + encoder + classifier head ---
    elif func_name.endswith("test_vit"):
        # Patch embedding
        ops += infer_vit_ops("test_vit_patch_embeddings", params, cfg, batch)
        # Encoder
        ops += infer_vit_ops("test_vit_encoder", params, cfg, batch)
        # Classifier head: (batch, H) x (H, num_classes=1000)
        ops.append(make(
            f"Classifier head H={H} num_classes=1000",
            M=1, K=H, N=1000, src=func_name
        ))

    return ops


# ---------------------------------------------------------------------------
# Generic shape-comment scraper (fallback for unknown functions)
# ---------------------------------------------------------------------------

SHAPE_RE = re.compile(
    r"\((\w+)[,\s]+(\w+)[,\s]+(\w+)(?:[,\s]+(\w+))?\)"  # (batch, M, K, N) patterns
)
PARAM_ASSIGN_RE = re.compile(
    r"(?:batch_size|batch|M|K|N|hidden_size|intermediate_size|sequence_size)\s*=\s*(\d+)"
)

def scrape_shapes_from_source(source: str) -> list[tuple]:
    """
    Very rough heuristic: find lines like
       torch_random((batch_size, sequence_size, config.hidden_size), ...)
    and pull out numeric literal assignments from context.
    Returns list of (M, K, N) tuples where we found them.
    """
    found = []
    # collect numeric assignments
    nums = {}
    for m in PARAM_ASSIGN_RE.finditer(source):
        # key = preceding word; not reliable but better than nothing
        pass
    return found


# ---------------------------------------------------------------------------
# Main analyzer
# ---------------------------------------------------------------------------

class TestFileAnalyzer:
    def __init__(self, path: Path, model_key: str = "vit-base"):
        self.path = path
        self.source = path.read_text()
        self.tree = ast.parse(self.source)
        self.model_key = model_key
        self.cfg = MODEL_CONFIGS.get(model_key, MODEL_CONFIGS["vit-base"])
        self.ops: list[OpEntry] = []
        self._id = 1

    def analyze(self):
        for node in ast.walk(self.tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            if not node.name.startswith("test_"):
                continue
            params = extract_parametrize(node)
            batch = params.get("batch_size", 8)
            if batch is None:
                batch = 8

            new_ops = infer_vit_ops(node.name, params, self.cfg, batch)
            if not new_ops:
                # Fallback: emit a placeholder so the function is visible
                new_ops = [OpEntry(
                    op_id=0,
                    description=f"[TODO] {node.name} — shape inference not implemented",
                    source_test=node.name,
                    tensor_a=TensorSpec(),
                    tensor_b=TensorSpec(),
                    output=TensorSpec(),
                )]

            for op in new_ops:
                op.op_id = self._id
                self._id += 1
            self.ops.extend(new_ops)

    def to_yaml_records(self) -> list[dict]:
        records = []
        for op in self.ops:
            a = {k: v for k, v in asdict(op.tensor_a).items() if v is not None}
            b = {k: v for k, v in asdict(op.tensor_b).items() if v is not None}
            out = asdict(op.output)
            out = {k: v for k, v in out.items()
                   if k in ("dtype", "memory_layout", "buffer_type") and v is not None}
            rec = {
                "op_id": op.op_id,
                "description": op.description,
                "source_test": op.source_test,
                "op_type": op.op_type,
                "tensor_a": a,
                "tensor_b": b,
                "output": out,
                "compute": asdict(op.compute),
                "flags": op.flags,
            }
            records.append(rec)
        return records


# ---------------------------------------------------------------------------
# Interactive / manual mode
# ---------------------------------------------------------------------------

def interactive_mode(out_path: Path):
    """
    Let the user type in shapes manually when static inference isn't enough.
    """
    print("\n=== Manual op entry mode ===")
    print("Enter one matmul op at a time. Press Ctrl-C to finish.\n")
    ops = []
    op_id = 1
    try:
        while True:
            print(f"--- Op {op_id} ---")
            desc = input("  description: ")
            batch = int(input("  batch_a (default 1): ") or 1)
            M = int(input("  M: "))
            K = int(input("  K: "))
            N = int(input("  N: "))
            batch_b = int(input("  batch_b (default 1): ") or 1)
            ops.append(OpEntry(
                op_id=op_id,
                description=desc,
                tensor_a=TensorSpec(batch=batch, M=M, K=K),
                tensor_b=TensorSpec(batch=batch_b, K=K, N=N),
                output=TensorSpec(),
            ))
            op_id += 1
    except (KeyboardInterrupt, EOFError):
        pass

    if ops:
        records = []
        for op in ops:
            a = {k: v for k, v in asdict(op.tensor_a).items() if v is not None}
            b = {k: v for k, v in asdict(op.tensor_b).items() if v is not None}
            out = {"dtype": "BFLOAT16", "memory_layout": "INTERLEAVED", "buffer_type": "DRAM"}
            records.append({
                "op_id": op.op_id,
                "description": op.description,
                "tensor_a": a,
                "tensor_b": b,
                "output": out,
                "compute": asdict(op.compute),
                "flags": op.flags,
            })
        _write_yaml(records, out_path)
        print(f"\nWrote {len(records)} ops to {out_path}")
    else:
        print("No ops entered.")


def _write_yaml(records: list[dict], out_path: Path):
    class _Dumper(yaml.Dumper):
        pass
    _Dumper.add_representer(
        bool,
        lambda dumper, data: dumper.represent_scalar("tag:yaml.org,2002:bool", "true" if data else "false")
    )
    with open(out_path, "w") as f:
        yaml.dump(records, f, Dumper=_Dumper, default_flow_style=False, sort_keys=False, allow_unicode=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Convert TT-Metal test file → simulator YAML")
    parser.add_argument("test_file", nargs="?", help="Path to test_*.py file")
    parser.add_argument("--model", default="vit-base",
                        choices=list(MODEL_CONFIGS.keys()),
                        help="Model config key (default: vit-base)")
    parser.add_argument("--out", default="ops.yaml", help="Output YAML path (default: ops.yaml)")
    parser.add_argument("--manual", action="store_true",
                        help="Skip file analysis; enter shapes interactively")
    parser.add_argument("--list-models", action="store_true",
                        help="Print known model configs and exit")
    args = parser.parse_args()

    if args.list_models:
        print("Known model configs:")
        for k, v in MODEL_CONFIGS.items():
            print(f"  {k}: hidden={v['hidden_size']} intermediate={v['intermediate_size']} "
                  f"heads={v['num_attention_heads']} layers={v['num_hidden_layers']}")
        return

    out_path = Path(args.out)

    if args.manual:
        interactive_mode(out_path)
        return

    if not args.test_file:
        parser.print_help()
        sys.exit(1)

    test_path = Path(args.test_file)
    if not test_path.exists():
        print(f"Error: {test_path} not found", file=sys.stderr)
        sys.exit(1)

    analyzer = TestFileAnalyzer(test_path, model_key=args.model)
    analyzer.analyze()
    records = analyzer.to_yaml_records()
    _write_yaml(records, out_path)

    print(f"Extracted {len(records)} ops from {test_path.name}")
    print(f"Output: {out_path}")

    # Print a quick summary table
    print(f"\n{'ID':>4}  {'M':>6}  {'K':>6}  {'N':>6}  Description")
    print("-" * 70)
    for r in records:
        a = r["tensor_a"]
        b = r["tensor_b"]
        M = a.get("M", "?")
        K = a.get("K", "?")
        N = b.get("N", "?")
        print(f"{r['op_id']:>4}  {str(M):>6}  {str(K):>6}  {str(N):>6}  {r['description'][:40]}")


if __name__ == "__main__":
    main()
 
