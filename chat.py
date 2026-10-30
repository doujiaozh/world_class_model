#!/usr/bin/env python
"""world_class_model 交互终端 —— 本地极速路由 + 云端超级大脑。

用法：
    python chat.py                 # 交互 REPL
    python chat.py --demo          # 演示 6 条典型问题（API Key 可有可无）
    python chat.py --ask "10 + 5"  # 单问模式
    python chat.py --mode solver|local|cloud|auto

环境变量：
    DEEPSEEK_API_KEY   云端大脑开关（设置后启用 DeepSeek API，不设则本地兜底）
    WCM_BASE_URL       自定义 OpenAI 兼容端点（默认 https://api.deepseek.com/v1）
    WCM_MODEL          模型名（默认 deepseek-chat）
    WCM_NO_STREAM      设为 1 关闭流式输出
"""

from __future__ import annotations

import argparse
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from wcm import router  # noqa: E402
from wcm.cloud import CloudError, available as cloud_available, chat as cloud_chat, preprocess  # noqa: E402

BANNER = r"""
 __        __   _     ___ _   ___ ___    _      __  __   _   ___ ___ _  __ ___ _   _ ___
 \ \      / /__| |__ | __| \ | __|_ _|  /_\    / / |  \ | | | __|_ _| |/ /| __| | | | _ \
  \ \/\/ / -_) '_ \(_-<|  \| _|__ | |  / _ \  / /  | \  | |_| _|__ | || ' < | _|| |_| |  _/
   \_/\_/\___|_.__//___|_|\_|___|___|_/_/ \_\/_/   |__|\___|___|___|_|\_\|___|\___/|_|

  混合超级引擎 v2.0 ——  本地极速路由（μs） + 本地微模型（ms） + 云端超级大脑
"""


def _c(code: str, text: str) -> str:
    if not sys.stdout.isatty():
        return text
    return f"\033[{code}m{text}\033[0m"


LOCAL_CACHE: dict = {}


def local_answer(prompt: str, ckpt: str = router.DEFAULT_CKPT, n_new: int = 24) -> str:
    """Tier-1 本地微模型生成兜底。"""
    if "params" not in LOCAL_CACHE:
        import numpy as np
        from wcm import model as model_mod
        from wcm.quantize import load_quantized
        from wcm.tokenizer import BPETokenizer
        from wcm.utils import load_checkpoint

        if ckpt.endswith("_int8.npz"):
            params, cfg, _ = load_quantized(ckpt)
        else:
            params, cfg, _ = load_checkpoint(ckpt)
        tok = BPETokenizer()
        tok.load(os.path.join(os.path.dirname(ckpt), "tokenizer.json"))
        LOCAL_CACHE.update(params=params, cfg=cfg, tok=tok, model=model_mod, np=np)
    m, np = LOCAL_CACHE["model"], LOCAL_CACHE["np"]
    ids = np.array(LOCAL_CACHE["tok"].encode(prompt)[-LOCAL_CACHE["cfg"].max_seq_len:],
                   dtype=np.int64)
    out = m.generate(LOCAL_CACHE["params"], LOCAL_CACHE["cfg"], ids[None, :],
                     n_new=n_new, temperature=0.0)
    return LOCAL_CACHE["tok"].decode(out[len(ids):])


def answer(text: str, mode: str = "auto", use_llm: bool = False,
           ckpt: str = router.DEFAULT_CKPT) -> tuple[str, dict]:
    """统一问答入口。返回 (答案, 路由信息)。"""
    t0 = time.perf_counter()

    def done(route: str, out: str, note: str = "") -> tuple[str, dict]:
        meta = {"route": route, "ms": (time.perf_counter() - t0) * 1000, "note": note}
        return out, meta

    if mode == "solver":
        res = router.solver_mod.solve_text(text)
        if res["ok"]:
            return done("Tier-0 求解器", f"{res['expr']} = {res['value_str']}")
        return done("Tier-0 求解器", "无法精确求解（不是可解析的算术表达式）", "fallback=none")

    if mode == "local":
        return done("Tier-1 本地微模型", local_answer(text, ckpt).strip() or "（本地模型无输出）")

    if mode == "cloud":
        return done("Tier-2 云端大脑", cloud_chat(preprocess(text)).strip())

    # ---- auto：混合路由 ----
    intent, info = router.decide(text, use_llm=use_llm, ckpt=ckpt)
    if intent == "calc" and info["solved"]:
        steps = "\n".join("    " + s for s in info["steps"])
        body = f"{info['expr']} = {info['value_str']}"
        if steps:
            body += "\n" + _c("90", "分步：\n" + steps)
        return done("Tier-0 求解器", body)

    if cloud_available():
        try:
            out = cloud_chat(preprocess(text)).strip()
            return done("Tier-2 云端大脑", out)
        except CloudError as e:
            note = f"云端失败，降级本地：{e}"
    else:
        note = "未配置 DEEPSEEK_API_KEY，本地微模型兜底（设置后体验世界级推理）"

    try:
        out = local_answer(text, ckpt).strip() or "（本地模型无输出）"
    except Exception as e:  # noqa: BLE001
        return done("Tier-1 本地微模型", f"本地引擎不可用：{e}", note)
    return done("Tier-1 本地微模型", out, note)


DEMO_PROMPTS = [
    "10 + 5",
    "999999999 * 999999999",
    "2 ^ 10 + 3 * 4",
    "帮我写一个 Python 快速排序函数",
    "分析一下 Transformer 注意力机制的优缺点",
    "你好，介绍一下你自己",
]


def run_demo(args) -> None:
    print(_c("96", "=== 混合超级引擎演示（路由过程完全公开） ===\n"))
    for p in DEMO_PROMPTS:
        out, meta = answer(p, mode=args.mode, use_llm=False, ckpt=args.ckpt)
        tag = f"[{meta['route']}]  {meta['ms']:.1f}ms"
        print(f"{_c('97', '提问')} {p}")
        print(f"{_c('90', tag)}" + (f" {_c('93', meta['note'])}" if meta["note"] else ""))
        print(f"{_c('92', '回答')} {out}\n")


def repl(args) -> None:
    print(_c("92", BANNER))
    print(f"  路由模式: {_c('96', args.mode)} | 云端大脑: "
          f"{_c('92', '已接入') if cloud_available() else _c('93', '未配置 Key（本地兜底）')}")
    print("  命令: /help /mode <auto|solver|local|cloud> /route <问题> /demo /exit\n")
    mode = args.mode
    while True:
        try:
            q = input(_c("96", "wcm> ")).strip()
        except (EOFError, KeyboardInterrupt):
            print("\n" + _c("92", "再见！"))
            break
        if not q:
            continue
        if q == "/exit":
            break
        if q == "/help":
            print("  /mode <auto|solver|local|cloud>  切换路由模式")
            print("  /route <问题>                    只看路由决策不看回答")
            print("  /demo                            演示")
            print("  /exit                            退出")
            continue
        if q.startswith("/mode "):
            m = q[6:].strip()
            if m in ("auto", "solver", "local", "cloud"):
                mode, _ = m, print(f"  路由模式 → {m}")
            else:
                print("  无效模式，可选: auto | solver | local | cloud")
            continue
        if q.startswith("/route "):
            intent, info = router.decide(q[7:].strip())
            print(f"  intent={intent} solved={info['solved']} expr={info['expr']}")
            continue
        if q == "/demo":
            run_demo(args)
            continue
        if q.startswith("/"):
            print("  未知命令，输入 /help")
            continue
        out, meta = answer(q, mode=mode, ckpt=args.ckpt)
        tag = f"[{meta['route']}]  {meta['ms']:.1f}ms"
        print(f"{_c('90', tag)}" + (f" {_c('93', meta['note'])}" if meta["note"] else ""))
        print(out + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description="world_class_model 混合超级引擎交互终端")
    ap.add_argument("--demo", action="store_true", help="演示模式")
    ap.add_argument("--ask", type=str, default=None, help="单问模式")
    ap.add_argument("--mode", choices=["auto", "solver", "local", "cloud"],
                    default="auto", help="路由模式（默认 auto）")
    ap.add_argument("--ckpt", default=router.DEFAULT_CKPT, help="本地微模型检查点")
    args = ap.parse_args()
    if args.demo:
        run_demo(args)
        return 0
    if args.ask is not None:
        out, meta = answer(args.ask, mode=args.mode, ckpt=args.ckpt)
        tag = f"[{meta['route']}]  {meta['ms']:.1f}ms"
        print(tag + (f" {meta['note']}" if meta["note"] else ""))
        print(out)
        return 0
    repl(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
