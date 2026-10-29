"""能力评测基准：用训练时未见过的随机种子生成新题，贪心解码，精确匹配判分。

跑法：python eval_bench.py --ckpt artifacts/wcm_tiny.npz --n 60
类别：加法 / 减法 / 乘法 / 比较 / 反转 —— 每类可量化、可复现。
"""
from __future__ import annotations

import argparse
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wcm.model import generate
from wcm.quantize import load_quantized
from wcm.tokenizer import BPETokenizer
from wcm.utils import load_checkpoint


def make_tasks(n: int, seed: int):
    rng = random.Random(seed)
    tasks = []
    for _ in range(n):
        a, b = rng.randint(10, 99), rng.randint(10, 99)
        if rng.random() < 0.5:
            tasks.append(("add", f"{a} + {b} =", f" {a + b}"))
        else:
            if a < b:
                a, b = b, a
            tasks.append(("sub", f"{a} - {b} =", f" {a - b}"))
    for _ in range(n):
        a, b = rng.randint(2, 12), rng.randint(2, 12)
        tasks.append(("mul", f"{a} x {b} =", f" {a * b}"))
    for _ in range(n):
        a, b = rng.randint(0, 99), rng.randint(0, 99)
        tasks.append(("compare", f"which is bigger {a} or {b} ?", f" {max(a, b)}"))
    for _ in range(n):
        k = rng.randint(3, 7)
        s = "".join(rng.choice("abcdef") for _ in range(k))
        tasks.append(("reverse", f"reverse ( {s} ) =", f" {s[::-1]}"))
    return tasks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="artifacts/wcm_tiny.npz")
    ap.add_argument("--tokenizer", default="artifacts/tokenizer.json")
    ap.add_argument("--n", type=int, default=40, help="每类题目数")
    ap.add_argument("--seed", type=int, default=999, help="出题种子（与训练集隔离）")
    ap.add_argument("--max_new", type=int, default=6)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    ckpt_path = os.path.join(here, args.ckpt)
    tok = BPETokenizer().load(os.path.join(here, args.tokenizer))
    if args.ckpt.endswith("_int8.npz"):
        params, cfg, _ = load_quantized(ckpt_path)
    else:
        params, cfg, _ = load_checkpoint(ckpt_path)

    tasks = make_tasks(args.n, args.seed)
    correct = {}
    total = {}
    examples = []
    for kind, prompt, expected in tasks:
        ids = tok.encode(prompt)
        out = generate(params, cfg, ids[None, :], args.max_new, temperature=0.0)
        produced = tok.decode(out[len(ids):])  # 只看生成部分
        ok = produced.startswith(expected)
        correct[kind] = correct.get(kind, 0) + int(ok)
        total[kind] = total.get(kind, 0) + 1
        if len(examples) < 8:
            examples.append((kind, prompt, expected.strip(), produced.strip(), ok))

    all_c = sum(correct.values())
    all_t = sum(total.values())
    tag = f"[{args.tag}] " if args.tag else ""
    for k in sorted(total):
        print(f"{tag}{k:8s}: {correct[k]:3d}/{total[k]:3d} = {correct[k]/total[k]*100:5.1f}%", flush=True)
    print(f"{tag}TOTAL    : {all_c:3d}/{all_t:3d} = {all_c/all_t*100:5.1f}%", flush=True)
    print("\n样本（prompt -> 期望 | 生成 | 对错）:")
    for kind, p, e, pr, ok in examples:
        print(f"  [{kind}] {p} -> {e} | {pr} | {'OK' if ok else 'X'}", flush=True)


if __name__ == "__main__":
    main()
