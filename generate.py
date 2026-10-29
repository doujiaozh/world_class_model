"""推理入口：python generate.py --ckpt artifacts/wcm_tiny.npz --prompt "12 + 7 =" """
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wcm.model import generate
from wcm.quantize import load_quantized
from wcm.tokenizer import BPETokenizer
from wcm.utils import load_checkpoint


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="artifacts/wcm_tiny.npz")
    ap.add_argument("--tokenizer", default="artifacts/tokenizer.json")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--n_new", type=int, default=12)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_k", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    ckpt_path = os.path.join(here, args.ckpt)
    tok = BPETokenizer().load(os.path.join(here, args.tokenizer))

    if args.ckpt.endswith("_int8.npz"):
        params, cfg, _ = load_quantized(ckpt_path)
    else:
        params, cfg, _ = load_checkpoint(ckpt_path)

    ids = tok.encode(args.prompt)
    out = generate(params, cfg, ids[None, :], args.n_new,
                   temperature=args.temperature, top_k=args.top_k, seed=args.seed)
    text = tok.decode(out)
    print(text, flush=True)


if __name__ == "__main__":
    main()
