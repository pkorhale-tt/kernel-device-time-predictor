# generate_square_sweep_yaml.py
# Usage: python generate_square_sweep_yaml.py --out square_sweep.yaml
'''
# default — full sweep 32 to 4096 step 32, batch 1 and 8, all 3 fidelities
python generate_square_sweep_yaml.py

# custom step size (fewer ops, faster)
python generate_square_sweep_yaml.py --step 64

# only HiFi2 to start
python generate_square_sweep_yaml.py --fids HiFi2 --out square_hifi2_only.yaml

# specific sizes only
python generate_square_sweep_yaml.py \
  --sizes 32,64,128,256,512,768,1024,2048,3072,4096 \
  --out square_sweep_keypoints.yaml

'''
import yaml
import argparse

# ── Config ────────────────────────────────────────────────────────────────────

SIZES    = list(range(32, 4097, 32))   # 32, 64, 96, ... 4096
BATCHES  = [1, 8]
FIDS     = ['LoFi', 'HiFi2', 'HiFi4']

# ── Generate ops ──────────────────────────────────────────────────────────────

def make_op(op_id, batch, S, fid):
    return {
        'op_id'      : op_id,
        'description': f"Matmul M={S} K={S} N={S} batch={batch} fid={fid}",
        'tensor_a'   : {
            'batch'          : batch,
            'M'              : S,
            'K'              : S,
            'dtype'          : 'BFLOAT16',
            'memory_layout'  : 'INTERLEAVED',
            'buffer_type'    : 'DRAM',
            'transpose'      : False,
        },
        'tensor_b'   : {
            'batch'          : 1,
            'K'              : S,
            'N'              : S,
            'dtype'          : 'BFLOAT16',
            'memory_layout'  : 'INTERLEAVED',
            'buffer_type'    : 'DRAM',
            'transpose'      : False,
        },
        'output'     : {
            'dtype'          : 'BFLOAT16',
            'memory_layout'  : 'INTERLEAVED',
            'buffer_type'    : 'DRAM',
        },
        'compute'    : {
            'math_fidelity'   : fid,
            'fp32_dest_acc_en': False,
            'packer_l1_acc'   : True,
            'math_approx_mode': False,
        },
        'flags'      : {
            'user_run_batched': False,
        },
    }


def main():
    p = argparse.ArgumentParser(
        description="Generate square matrix sweep YAML")
    p.add_argument('--out',    default='square_sweep.yaml',
                   help='Output YAML file (default: square_sweep.yaml)')
    p.add_argument('--sizes',  default=None,
                   help='Comma separated sizes e.g. 32,64,128 (default: 32..4096 step 32)')
    p.add_argument('--batches',default='1,8',
                   help='Comma separated batch values (default: 1,8)')
    p.add_argument('--fids',   default='LoFi,HiFi2,HiFi4',
                   help='Comma separated fidelities (default: LoFi,HiFi2,HiFi4)')
    p.add_argument('--step',   default=32, type=int,
                   help='Step size between matrix sizes (default: 32)')
    args = p.parse_args()

    # parse args
    if args.sizes:
        sizes = [int(s) for s in args.sizes.split(',')]
    else:
        sizes = list(range(32, 4097, args.step))

    batches = [int(b) for b in args.batches.split(',')]
    fids    = [f.strip() for f in args.fids.split(',')]

    # generate ops
    ops    = []
    op_id  = 1

    for batch in batches:
        for S in sizes:
            for fid in fids:
                ops.append(make_op(op_id, batch, S, fid))
                op_id += 1

    total = len(ops)

    # build workload dict
    workload = {
        'workload': {
            'name'       : 'square_sweep',
            'total_ops'  : total,
            'description': (
                f"Square matrix sweep "
                f"S={sizes[0]}..{sizes[-1]} step={args.step} "
                f"batches={batches} "
                f"fidelities={fids}"
            ),
            'sizes'      : sizes,
            'batches'    : batches,
            'fidelities' : fids,
            'ops'        : ops,
        }
    }

    # custom dumper — keeps ops readable
    class _Dumper(yaml.Dumper):
        pass

    _Dumper.add_representer(
        bool,
        lambda d, v: d.represent_scalar(
            'tag:yaml.org,2002:bool', 'true' if v else 'false')
    )

    with open(args.out, 'w') as f:
        yaml.dump(workload, f,
                  Dumper=_Dumper,
                  default_flow_style=False,
                  sort_keys=False,
                  allow_unicode=True)

    print(f"Generated {total} ops → {args.out}")
    print(f"  sizes    : {sizes[0]} to {sizes[-1]} step {args.step} ({len(sizes)} sizes)")
    print(f"  batches  : {batches}")
    print(f"  fidelity : {fids}")
    print(f"  total    : {len(sizes)} × {len(batches)} × {len(fids)} = {total} ops")


if __name__ == '__main__':
    main()
