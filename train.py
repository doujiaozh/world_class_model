"""训练入口：python train.py --config tiny --steps 500"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np

from wcm.config import Config, count_params, get_config
from wcm.data import batch_stream
from wcm.model import init_params, forward, backward
from wcm.optimizer import AdamW
from wcm.quantize import save_quantized
from wcm.tokenizer import BPETokenizer
from wcm.utils import save_checkpoint, load_checkpoint


def maybe_train_tokenizer(tok_path: str, corpus_path: str, vocab_size: int) -> BPETokenizer:
    tok = BPETokenizer()
    if os.path.exists(tok_path):
        tok.load(tok_path)
        print(f"加载已有分词器 {tok_path}（词表 {tok.vocab_size}）", flush=True)
    else:
        with open(corpus_path, encoding="utf-8") as f:
            text = f.read()
        t0 = time.time()
        tok.train(text, vocab_size, verbose=True)
        os.makedirs(os.path.dirname(tok_path), exist_ok=True)
        tok.save(tok_path)
        print(f"分词器训练耗时 {time.time()-t0:.1f}s", flush=True)
    return tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="tiny")
    ap.add_argument("--corpus", default=os.path.join("data", "corpus.txt"))
    ap.add_argument("--tokenizer", default="artifacts/tokenizer.json")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seq_len", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    ap.add_argument("--resume", default="")
    ap.add_argument("--eval_every", type=int, default=50)
    ap.add_argument("--save_every", type=int, default=0, help="每 N 步存一次中间检查点（0=不存）")
    args = ap.parse_args()

    cfg = get_config(args.config)
    here = os.path.dirname(os.path.abspath(__file__))
    corpus_path = os.path.join(here, args.corpus)
    tok_path = os.path.join(here, args.tokenizer)
    out = args.out or os.path.join(here, "artifacts", f"wcm_{cfg.name}.npz")

    tok = maybe_train_tokenizer(tok_path, corpus_path, cfg.vocab_size)
    if tok.vocab_size != cfg.vocab_size:
        # 合成语料字符空间有限，BPE 达不到预设词表时按分词器实际词表对齐
        print(f"分词器实际词表 {tok.vocab_size}，模型配置由 {cfg.vocab_size} 对齐调整", flush=True)
        cfg.vocab_size = tok.vocab_size

    rng = np.random.default_rng(args.seed)
    if args.resume:
        params, cfg_load, _ = load_checkpoint(os.path.join(here, args.resume))
        print(f"从 {args.resume} 恢复参数", flush=True)
    else:
        params = init_params(cfg, seed=args.seed)
    print(f"配置 {cfg.name}：参数量 {count_params(cfg)/1e6:.2f}M", flush=True)

    # 语料 tokenize（懒加载已按词缓存，一次完成）
    t0 = time.time()
    from wcm.data import load_ids
    ids = load_ids(corpus_path, tok)
    print(f"语料 {len(ids)/1e3:.0f}k tokens，tokenize 耗时 {time.time()-t0:.1f}s", flush=True)

    stream = batch_stream(ids, args.batch, args.seq_len, rng)
    opt = AdamW(params, lr=args.lr)

    running = []
    t0 = time.time()
    for step in range(1, args.steps + 1):
        # 注意：不能把 lr 赋回 opt.lr，lr_at 以 self.lr 为基数，否则会逐步自乘坍塌
        lr = opt.lr_at(step, args.steps, warmup=args.warmup)
        x, y = next(stream)
        logits, loss, tape = forward(params, cfg, x, targets=y)
        grads = backward(params, cfg, tape, tape["dlogits"])
        opt.step(params, grads, step, lr=lr)
        running.append(loss)
        if args.save_every and step % args.save_every == 0:
            mid = out.replace(".npz", f"_step{step}.npz")
            save_checkpoint(mid, params, cfg, extra={"step": step, "train_loss": float(loss)})
            print(f"  [存盘] {mid}（{os.path.getsize(mid)/1e6:.1f}MB）", flush=True)
        if step % args.eval_every == 0 or step == 1:
            dt = (time.time() - t0) / step
            print(f"step {step:5d} | loss {np.mean(running[-args.eval_every:]):.4f} "
                  f"| lr {lr:.2e} | {dt:.2f}s/step", flush=True)

    os.makedirs(os.path.dirname(out), exist_ok=True)
    save_checkpoint(out, params, cfg, extra={"train_loss": float(np.mean(running)), "steps": args.steps})
    qsize = save_quantized(out.replace(".npz", "_int8.npz"), params, cfg,
                           meta={"origin": os.path.basename(out)})
    print(f"已保存检查点 {out}（fp32 {os.path.getsize(out)/1e6:.1f}MB）"
          f" 与 int8 量化版（{qsize/1e6:.1f}MB）", flush=True)
    print(f"最终训练 loss: {np.mean(running[-100:]):.4f}", flush=True)


if __name__ == "__main__":
    main()
