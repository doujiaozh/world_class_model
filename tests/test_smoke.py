"""冒烟与正确性测试：python tests/test_smoke.py

核心手段：对每个手写反向算子做中心差分数值梯度校验（无任何自动微分，
反传正确性靠数学推导 + 数值验证双重保证）。另含分词器可逆性、
KV cache 与训练前向的一致性、int8 量化误差界。
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wcm.config import Config
from wcm.model import (
    _attn_bwd, _attn_fwd, _ffn_bwd, _ffn_fwd, backward, forward,
    forward_step, generate, init_params, rmsnorm_fwd, rmsnorm_grad,
    rope_fwd, rope_grad, rope_tables, silu_grad,
)
from wcm.quantize import dequantize_params, quantize_params
from wcm.tokenizer import BPETokenizer

CFG = Config(name="test", vocab_size=64, d_model=32, n_layers=2, n_heads=4,
             n_kv_heads=2, ffn_hidden=48, max_seq_len=32)
B, T = 2, 6


def num_grad(loss_fn, x, h=1e-3, check=6, rng=None):
    """中心差分：loss_fn(x) -> 标量。随机抽 check 个分量验证。"""
    rng = rng or np.random.default_rng(0)
    flat = x.ravel()
    grads = np.zeros_like(flat, dtype=np.float64)
    idxs = rng.choice(flat.size, size=min(check, flat.size), replace=False)
    for i in idxs:
        old = flat[i]
        flat[i] = old + h
        lp = loss_fn()
        flat[i] = old - h
        lm = loss_fn()
        flat[i] = old
        grads[i] = (lp - lm) / (2 * h)
    return grads.reshape(x.shape), idxs


def rel_err(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    denom = np.maximum(np.abs(a) + np.abs(b), 1e-12)
    return float(np.max(np.abs(a - b) / denom))


def match(a, b, rtol=1e-3, atol=1e-3):
    """梯度匹配判据：相对/绝对双判据（float32 下小梯度不可用纯相对误差）。"""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return bool(np.all(np.abs(a - b) <= atol + rtol * np.abs(b)))


def test_rmsnorm(rng):
    x = rng.normal(size=(4, 5, 6)).astype(np.float32)
    w = rng.normal(size=6).astype(np.float32)
    R = rng.normal(size=(4, 5, 6)).astype(np.float32)
    y, r = rmsnorm_fwd(x, w, 1e-5)

    def loss():
        return float((R * rmsnorm_fwd(x, w, 1e-5)[0]).sum())

    dx, dw = rmsnorm_grad(R, x, w, r)
    nx, ixs = num_grad(loss, x, check=5, rng=rng)
    ok_x = match(dx.ravel()[ixs], nx.ravel()[ixs])
    nw, jxs = num_grad(loss, w, check=5, rng=rng)
    return ok_x and match(dw[jxs], nw[jxs])


def test_rope(rng):
    x = rng.normal(size=(2, 5, 3, 8)).astype(np.float32)
    cos, sin = rope_tables(CFG, 5)
    R = rng.normal(size=x.shape).astype(np.float32)

    def loss():
        return float((R * rope_fwd(x, cos, sin)).sum())

    g = rope_grad(R, cos, sin)
    ng, idxs = num_grad(loss, x, check=5, rng=rng)
    return match(g.ravel()[idxs], ng.ravel()[idxs])


def test_silu(rng):
    x = rng.normal(size=(50,)).astype(np.float32)
    g = silu_grad(x)
    h = 1e-2
    ng = ((x + h) / (1 + np.exp(-(x + h))) - (x - h) / (1 + np.exp(-(x - h)))) / (2 * h)
    return match(g, ng, rtol=1e-3, atol=1e-3)


def test_attn(rng):
    from wcm.config import Config as C
    cfg = C(name="t", vocab_size=8, d_model=16, n_layers=1, n_heads=4,
            n_kv_heads=2, ffn_hidden=32, max_seq_len=16)
    params = init_params(cfg, seed=1)
    x1 = rng.normal(size=(B, T, 16)).astype(np.float32)
    cos, sin = rope_tables(cfg, T)
    out, cache = _attn_fwd(x1, params, cfg, cos, sin, 0)
    R = rng.normal(size=out.shape).astype(np.float32)
    dx1, grads = _attn_bwd(R, cache, params, cfg, cos, sin, 0)

    def loss():
        o, _ = _attn_fwd(x1, params, cfg, cos, sin, 0)
        return float((R * o).sum())

    ok = True
    nx, idxs = num_grad(loss, x1, check=5, rng=rng)
    ok &= match(dx1.ravel()[idxs], nx.ravel()[idxs], rtol=1e-2, atol=1e-2)
    for key in ["attn_q_w", "attn_k_w", "attn_v_w", "attn_o_w"]:
        w = params[f"l0.{key}"]
        nw, idxs = num_grad(loss, w, check=4, rng=rng)
        ok &= match(grads[f"l0.{key}"].ravel()[idxs], nw.ravel()[idxs], rtol=1e-2, atol=1e-2)
    return ok


def test_ffn(rng):
    params = init_params(CFG, seed=1)
    x2 = rng.normal(size=(B, T, CFG.d_model)).astype(np.float32)
    out, cache = _ffn_fwd(x2, params, 0)
    R = rng.normal(size=out.shape).astype(np.float32)
    dx2, grads = _ffn_bwd(R, cache, params, 0)

    def loss():
        o, _ = _ffn_fwd(x2, params, 0)
        return float((R * o).sum())

    ok = True
    nx, idxs = num_grad(loss, x2, check=5, rng=rng)
    ok &= match(dx2.ravel()[idxs], nx.ravel()[idxs], rtol=1e-2, atol=1e-2)
    for key in ["ffn_gate_w", "ffn_up_w", "ffn_down_w"]:
        w = params[f"l0.{key}"]
        nw, idxs = num_grad(loss, w, check=4, rng=rng)
        ok &= match(grads[f"l0.{key}"].ravel()[idxs], nw.ravel()[idxs], rtol=1e-2, atol=1e-2)
    return ok


def test_full_model(rng):
    params = init_params(CFG, seed=1)
    idx = rng.integers(0, CFG.vocab_size, size=(B, T)).astype(np.int64)
    y = rng.integers(0, CFG.vocab_size, size=(B, T)).astype(np.int64)
    logits, loss, tape = forward(params, CFG, idx, targets=y)
    grads = backward(params, CFG, tape, tape["dlogits"])

    def loss_fn():
        _, l, _ = forward(params, CFG, idx, targets=y)
        return l

    keys = ["tok_emb", "ln_f_w", "l0.ln1_w", "l0.attn_k_w", "l0.attn_q_w",
            "l0.ffn_down_w", "l1.ln2_w", "l1.ffn_gate_w"]
    ok = True
    for k in keys:
        w = params[k]
        ng, idxs = num_grad(loss_fn, w, h=1e-2, check=4, rng=rng)
        ok &= match(grads[k].ravel()[idxs], ng.ravel()[idxs], rtol=2e-2, atol=2e-2)
    return ok


def test_kv_cache(rng):
    params = init_params(CFG, seed=1)
    idx = rng.integers(0, CFG.vocab_size, size=(1, 9)).astype(np.int64)
    logits_full, _, _ = forward(params, CFG, idx)
    # 增量推理
    from wcm.model import new_cache
    cache = new_cache(CFG, 1)
    step_logits = []
    for t in range(idx.shape[1]):
        lg = forward_step(params, CFG, idx[:, t:t + 1], cache)
        step_logits.append(lg[0, -1])
    step_logits = np.stack(step_logits)
    return match(step_logits, logits_full[0], rtol=1e-2, atol=1e-2)


def test_generate_shape(rng):
    params = init_params(CFG, seed=1)
    out = generate(params, CFG, np.asarray([[1, 2, 3]]), n_new=4, seed=0)
    return out.shape == (7,)


def test_tokenizer_roundtrip():
    text = "hello world\n12 + 34 = 56\nreverse ( abc ) = cba\n中文也要行 ✓"
    tok = BPETokenizer()
    tok.train(text * 20, vocab_size=400)
    ids = tok.encode(text)
    back = tok.decode(ids)
    assert back == text, f"round-trip 失败: {back!r} != {text!r}"
    assert tok.vocab_size == 256 + len(tok.merges)
    # 新文本可编码（含未登录子词切回字节）
    ids2 = tok.encode("unseen_xyz 999")
    assert len(ids2) > 0 and tok.decode(ids2) == "unseen_xyz 999"
    return True


def test_quantize():
    rng = np.random.default_rng(0)
    params = {"w": rng.normal(size=(64, 32)).astype(np.float32),
              "v": np.ones(64, dtype=np.float32)}
    q = quantize_params(params)
    de = dequantize_params(q)
    # 正确判据：逐行对称 int8 量化的重构误差 ≤ 行最大值 / 127（小元素相对误差天然大）
    row_max = np.max(np.abs(params["w"]), axis=1, keepdims=True)
    ok = bool(np.all(np.abs(de["w"] - params["w"]) <= row_max / 127.0 + 1e-6))
    assert ok, "量化重构误差超出逐行上界"
    assert np.allclose(de["v"], 1.0)
    assert de["w"].dtype == np.float32 and q["w"].dtype == np.int8
    return True


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    tests = [
        ("rmsnorm 反向传播", lambda: test_rmsnorm(rng)),
        ("RoPE 反向传播", lambda: test_rope(rng)),
        ("SiLU 导数", lambda: test_silu(rng)),
        ("注意力块反向传播", lambda: test_attn(rng)),
        ("SwiGLU FFN 反向传播", lambda: test_ffn(rng)),
        ("完整模型反向传播", lambda: test_full_model(rng)),
        ("KV cache 一致性", lambda: test_kv_cache(rng)),
        ("generate 输出形状", lambda: test_generate_shape(rng)),
        ("BPE 编解码可逆", test_tokenizer_roundtrip),
        ("int8 量化误差界", test_quantize),
    ]
    allok = True
    for name, fn in tests:
        try:
            ok = fn()
        except AssertionError as e:
            ok, msg = False, str(e)
        else:
            msg = ""
        status = "PASS" if ok else "FAIL"
        allok &= bool(ok)
        print(f"[{status}] {name}" + (f"  ({msg})" if msg else ""), flush=True)
    print("全部通过" if allok else "存在失败", flush=True)
    sys.exit(0 if allok else 1)
