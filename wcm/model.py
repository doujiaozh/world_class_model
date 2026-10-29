"""纯 NumPy 手写的解码器 Transformer。

架构（当代小模型标准件，全部自己实现）：
  - RMSNorm（无 bias）
  - RoPE 旋转位置编码（作用于 Q/K，推理时可增量）
  - 分组查询注意力 GQA（n_kv_heads < n_heads）
  - SwiGLU 前馈网络
  - 词嵌入与 LM head 权重共享（tied embedding）
  - 因果掩码；训练用含反传 tape 的 forward，推理用带 KV cache 的 forward_step

前向、反向全部手工推导实现（见 docstring 中的公式），无任何自动微分框架。
"""
from __future__ import annotations

import numpy as np

from .config import Config
from .utils import softmax


# ============================== 初始化 ==============================
def init_params(cfg: Config, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    d, h, dkv = cfg.d_model, cfg.ffn_hidden, cfg.d_kv
    p = {"tok_emb": rng.normal(0, 0.02, (cfg.vocab_size, d)).astype(np.float32)}
    for i in range(cfg.n_layers):
        p[f"l{i}.ln1_w"] = np.ones(d, dtype=np.float32)
        p[f"l{i}.attn_q_w"] = rng.normal(0, 1 / np.sqrt(d), (d, d)).astype(np.float32)
        p[f"l{i}.attn_k_w"] = rng.normal(0, 1 / np.sqrt(d), (d, dkv)).astype(np.float32)
        p[f"l{i}.attn_v_w"] = rng.normal(0, 1 / np.sqrt(d), (d, dkv)).astype(np.float32)
        p[f"l{i}.attn_o_w"] = rng.normal(0, 1 / np.sqrt(d), (d, d)).astype(np.float32)
        p[f"l{i}.ln2_w"] = np.ones(d, dtype=np.float32)
        p[f"l{i}.ffn_gate_w"] = rng.normal(0, 1 / np.sqrt(d), (d, h)).astype(np.float32)
        p[f"l{i}.ffn_up_w"] = rng.normal(0, 1 / np.sqrt(d), (d, h)).astype(np.float32)
        p[f"l{i}.ffn_down_w"] = rng.normal(0, 1 / np.sqrt(h), (h, d)).astype(np.float32)
    p["ln_f_w"] = np.ones(d, dtype=np.float32)
    return p


# ============================== 基础算子 ==============================
def rope_tables(cfg: Config, T: int, offset: int = 0):
    """返回 (cos, sin)，形状 (T, 1, head_dim//2)。

    中间的 1 维是为了和 (B, T, H, hd//2) 布局广播（1 对应头维）；
    offset 平移用于增量解码。
    """
    hd = cfg.head_dim
    inv = cfg.rope_base ** (-np.arange(0, hd, 2, dtype=np.float64) / hd)
    pos = np.arange(offset, offset + T, dtype=np.float64)
    ang = np.outer(pos, inv)
    return np.cos(ang).astype(np.float32)[:, None, :], np.sin(ang).astype(np.float32)[:, None, :]


def rmsnorm_fwd(x: np.ndarray, w: np.ndarray, eps: float):
    """y = x * w / rms(x)，rms = sqrt(mean(x^2) + eps)。返回 (y, r)。"""
    r = np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    return (x * w) / r, r


def rmsnorm_grad(g: np.ndarray, x: np.ndarray, w: np.ndarray, r: np.ndarray):
    """rmsnorm 反向传播。

    推导：y_i = x_i w_i / r，r 只依赖该 token 的 x。
      dx_i = g_i w_i / r - (Σ_j g_j y_j) * x_i / (n r^2)
      dw_i = Σ_tokens g_i x_i / r
    """
    y = (x * w) / r
    s = np.sum(g * y, axis=-1, keepdims=True)
    n = x.shape[-1]
    dx = (g * w - (s / n) * (x / r)) / r
    dw = np.sum(g * x / r, axis=tuple(range(x.ndim - 1)))
    return dx, dw


def rope_fwd(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    """对最后一维的 (even, odd) 对施加旋转。x: (..., T, hd)；cos/sin: (T, 1, hd//2)。"""
    xe, xo = x[..., 0::2], x[..., 1::2]
    out = np.empty_like(x)
    out[..., 0::2] = xe * cos - xo * sin
    out[..., 1::2] = xe * sin + xo * cos
    return out


def rope_grad(g: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    """RoPE 反向传播：旋转矩阵正交，反向 = 转置（反向旋转）。g: (..., T, hd)。"""
    ge, go = g[..., 0::2], g[..., 1::2]
    out = np.empty_like(g)
    out[..., 0::2] = ge * cos + go * sin
    out[..., 1::2] = -ge * sin + go * cos
    return out


def silu_fwd(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def silu_grad(x: np.ndarray) -> np.ndarray:
    s = 1.0 / (1.0 + np.exp(-x))
    return s * (1.0 + x * (1.0 - s))


# ============================== 注意力 ==============================
def _attn_fwd(x1, p, cfg, cos, sin, i):
    B, T, d = x1.shape
    H, Hkv, hd, rep = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, cfg.n_heads // cfg.n_kv_heads
    q = x1 @ p[f"l{i}.attn_q_w"]
    k = x1 @ p[f"l{i}.attn_k_w"]
    v = x1 @ p[f"l{i}.attn_v_w"]
    qh = rope_fwd(q.reshape(B, T, H, hd), cos, sin)
    kh = rope_fwd(k.reshape(B, T, Hkv, hd), cos, sin)
    kr = np.repeat(kh, rep, axis=2)
    vr = np.repeat(v.reshape(B, T, Hkv, hd), rep, axis=2)
    scale = 1.0 / np.sqrt(hd)
    # 张量布局统一为 (B, T, H, hd)；einsum 标签 t=序列位、h=头
    S = np.einsum("bthd,bshd->bhts", qh, kr) * scale   # (B,H,T,T)
    causal = np.triu(np.ones((T, T), dtype=bool), 1)
    S = np.where(causal, np.float32(-1e30), S)
    P = softmax(S, axis=-1)
    o = np.einsum("bhts,bshd->bthd", P, vr).reshape(B, T, d)
    cache = (x1, qh, kr, vr, P)
    return o @ p[f"l{i}.attn_o_w"], cache


def _attn_bwd(do2, cache, p, cfg, cos, sin, i):
    """do2: (B,T,d) 对注意力输出（投影后）的梯度。返回 (dx1, grads)。"""
    x1, qh, kr, vr, P = cache
    B, T, d = x1.shape
    H, Hkv, hd, rep = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, cfg.n_heads // cfg.n_kv_heads
    scale = 1.0 / np.sqrt(hd)
    Wq, Wk, Wv, Wo = (p[f"l{i}.attn_q_w"], p[f"l{i}.attn_k_w"],
                      p[f"l{i}.attn_v_w"], p[f"l{i}.attn_o_w"])
    o1 = np.einsum("bhts,bshd->bthd", P, vr).reshape(B, T, d)
    dWo = o1.reshape(-1, d).T @ do2.reshape(-1, d)
    do1h = (do2 @ Wo.T).reshape(B, T, H, hd)
    # softmax 反向 + 因果掩码（P=0 处自动归零）
    dP_raw = np.einsum("bthd,bshd->bhts", do1h, vr)     # (B,H,T,T)
    dS = P * (dP_raw - np.sum(dP_raw * P, axis=-1, keepdims=True))
    dvr = np.einsum("bhts,bthd->bshd", P, do1h)          # (B,T,H,hd)
    dq = np.einsum("bhts,bshd->bthd", dS, kr) * scale    # (B,T,H,hd)
    dkr = np.einsum("bhts,bthd->bshd", dS, qh) * scale   # (B,T,H,hd)
    # GQA：把重复头组的梯度沿组维相加
    dkv = dkr.reshape(B, T, Hkv, rep, hd).sum(axis=3)
    dvv = dvr.reshape(B, T, Hkv, rep, hd).sum(axis=3)
    # RoPE 反向（Q/K 都施加过）
    dq = rope_grad(dq, cos, sin).reshape(B, T, d)
    dk = rope_grad(dkv, cos, sin).reshape(B, T, Hkv * hd)
    dv = dvv.reshape(B, T, Hkv * hd)
    dWq = x1.reshape(-1, d).T @ dq.reshape(-1, d)
    dWk = x1.reshape(-1, d).T @ dk.reshape(-1, Hkv * hd)
    dWv = x1.reshape(-1, d).T @ dv.reshape(-1, Hkv * hd)
    dx1 = dq @ Wq.T + dk @ Wk.T + dv @ Wv.T
    grads = {f"l{i}.attn_q_w": dWq, f"l{i}.attn_k_w": dWk,
             f"l{i}.attn_v_w": dWv, f"l{i}.attn_o_w": dWo}
    return dx1, grads


# ============================== SwiGLU FFN ==============================
def _ffn_fwd(x2, p, i):
    g = x2 @ p[f"l{i}.ffn_gate_w"]
    u = x2 @ p[f"l{i}.ffn_up_w"]
    h = silu_fwd(g) * u
    return h @ p[f"l{i}.ffn_down_w"], (x2, g, u, h)


def _ffn_bwd(dy, cache, p, i):
    x2, g, u, h = cache
    d, hh = x2.shape[-1], h.shape[-1]
    dWd = h.reshape(-1, hh).T @ dy.reshape(-1, d)
    dh = dy @ p[f"l{i}.ffn_down_w"].T
    dg = dh * u * silu_grad(g)
    du = dh * silu_fwd(g)
    dWg = x2.reshape(-1, d).T @ dg.reshape(-1, hh)
    dWu = x2.reshape(-1, d).T @ du.reshape(-1, hh)
    dx2 = dg @ p[f"l{i}.ffn_gate_w"].T + du @ p[f"l{i}.ffn_up_w"].T
    grads = {f"l{i}.ffn_gate_w": dWg, f"l{i}.ffn_up_w": dWu, f"l{i}.ffn_down_w": dWd}
    return dx2, grads


# ============================== 完整模型（训练） ==============================
def forward(params: dict, cfg: Config, idx: np.ndarray, targets: np.ndarray | None = None):
    """前向传播（训练用，保存反传 tape）。

    idx: (B, T) int64；targets: (B, T) 已左移一位的目标 token。
    返回 (logits, loss, tape)；无 targets 时 loss=None。
    """
    B, T = idx.shape
    E = params["tok_emb"]
    x = E[idx]                                   # (B, T, d)
    cos, sin = rope_tables(cfg, T)
    tape = {"idx": idx, "layers": [], "cos": cos, "sin": sin}
    for i in range(cfg.n_layers):
        x0 = x                                        # 层输入（反传时 rmsnorm 需要）
        ln1w = params[f"l{i}.ln1_w"]
        x1, r1 = rmsnorm_fwd(x0, ln1w, cfg.rms_eps)
        ao, ac = _attn_fwd(x1, params, cfg, cos, sin, i)
        b = x0 + ao                                   # 残差 + FFN 前的输入
        ln2w = params[f"l{i}.ln2_w"]
        x2, r2 = rmsnorm_fwd(b, ln2w, cfg.rms_eps)
        fo, fc = _ffn_fwd(x2, params, i)
        x = b + fo
        tape["layers"].append((x0, x1, r1, ln1w, ac, b, r2, ln2w, fc))
    xf, rf = rmsnorm_fwd(x, params["ln_f_w"], cfg.rms_eps)
    logits = xf @ E.T                            # (B, T, V)，权重共享
    tape["xf"], tape["rf"], tape["xn"] = xf, rf, x

    loss, dlogits = None, None
    if targets is not None:
        z = logits - logits.max(axis=-1, keepdims=True)
        ez = np.exp(z)
        logz = z - np.log(ez.sum(axis=-1, keepdims=True))
        mb = np.arange(B)[:, None]
        mt = np.arange(T)[None, :]
        nll = -logz[mb, mt, targets]
        loss = float(nll.mean())
        dlogits = ez / ez.sum(axis=-1, keepdims=True)
        dlogits[mb, mt, targets] -= 1.0
        dlogits /= B * T
    tape["dlogits"] = dlogits
    return logits, loss, tape


def backward(params: dict, cfg: Config, tape: dict, dlogits: np.ndarray) -> dict:
    """反向传播，返回与 params 同 key 的梯度字典。"""
    B, T = dlogits.shape[:2]
    d = cfg.d_model
    V = cfg.vocab_size
    E = params["tok_emb"]
    g: dict = {}
    xf, rf, xn = tape["xf"], tape["rf"], tape["xn"]
    dxf = dlogits @ E                                    # (B,T,d)
    g["tok_emb"] = dlogits.reshape(-1, V).T @ xf.reshape(-1, d)
    dx, dln_f = rmsnorm_grad(dxf, xn, params["ln_f_w"], rf)
    g["ln_f_w"] = dln_f
    cos, sin = tape["cos"], tape["sin"]
    for i in range(cfg.n_layers - 1, -1, -1):
        layer = tape["layers"][i]
        x, x1, r1, ln1w, ac, b, r2, ln2w, fc = layer
        # ---- FFN 分支 ----
        dfo = dx                                         # 残差
        dx2, fgrads = _ffn_bwd(dfo, fc, params, i)
        db_from_norm, g[f"l{i}.ln2_w"] = rmsnorm_grad(dx2, b, ln2w, r2)
        db = dx + db_from_norm
        # ---- 注意力分支 ----
        dao = db
        dx1, agrads = _attn_bwd(dao, ac, params, cfg, cos, sin, i)
        da_from_norm, g[f"l{i}.ln1_w"] = rmsnorm_grad(dx1, x, ln1w, r1)
        dx = db + da_from_norm
        g.update(agrads)
        g.update(fgrads)
    # 输入端词嵌入梯度
    np.add.at(g["tok_emb"], tape["idx"], dx)
    return g


# ============================== 推理（KV cache） ==============================
def new_cache(cfg: Config, B: int) -> list[dict]:
    return [ {"k": None, "v": None} for _ in range(cfg.n_layers) ]


def forward_step(params: dict, cfg: Config, idx: np.ndarray, cache: list[dict]):
    """单 token 增量前向。idx: (B, 1)。自动用 cache 长度作为 RoPE 位置。"""
    B, T = idx.shape
    assert T == 1, "forward_step 每次只处理一个 token"
    E = params["tok_emb"]
    pos = 0 if cache[0]["k"] is None else cache[0]["k"].shape[1]
    cos, sin = rope_tables(cfg, 1, pos)
    x = E[idx]
    for i in range(cfg.n_layers):
        H, Hkv, hd, rep = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, cfg.n_heads // cfg.n_kv_heads
        x1, _ = rmsnorm_fwd(x, params[f"l{i}.ln1_w"], cfg.rms_eps)
        q = rope_fwd((x1 @ params[f"l{i}.attn_q_w"]).reshape(B, 1, H, hd), cos, sin)
        k = rope_fwd((x1 @ params[f"l{i}.attn_k_w"]).reshape(B, 1, Hkv, hd), cos, sin)
        v = (x1 @ params[f"l{i}.attn_v_w"]).reshape(B, 1, Hkv, hd)
        if cache[i]["k"] is None:
            cache[i]["k"], cache[i]["v"] = k, v
        else:
            cache[i]["k"] = np.concatenate([cache[i]["k"], k], axis=1)
            cache[i]["v"] = np.concatenate([cache[i]["v"], v], axis=1)
        kr = np.repeat(cache[i]["k"], rep, axis=2)
        vr = np.repeat(cache[i]["v"], rep, axis=2)
        # 布局 (B,T,H,hd)，与 _attn_fwd 一致
        S = np.einsum("bthd,bshd->bhts", q, kr) / np.sqrt(hd)
        P = softmax(S, axis=-1)
        o = np.einsum("bhts,bshd->bthd", P, vr).reshape(B, 1, cfg.d_model)
        b = x + o @ params[f"l{i}.attn_o_w"]
        x2, _ = rmsnorm_fwd(b, params[f"l{i}.ln2_w"], cfg.rms_eps)
        x = b + _ffn_fwd(x2, params, i)[0]
    xf, _ = rmsnorm_fwd(x, params["ln_f_w"], cfg.rms_eps)
    return xf @ E.T                                     # (B, 1, V)


def generate(params: dict, cfg: Config, prompt_ids: np.ndarray, n_new: int,
             temperature: float = 1.0, top_k: int = 0, seed: int = 0) -> np.ndarray:
    """贪心或采样生成 n_new 个 token。prompt_ids: (1, T_prompt)。"""
    rng = np.random.default_rng(seed)
    B, T0 = prompt_ids.shape
    cache = new_cache(cfg, B)
    out = list(prompt_ids[0])
    # 先把 prompt 逐 token 灌进 cache（复用同一套增量前向）
    logits = None
    for t in range(T0):
        logits = forward_step(params, cfg, prompt_ids[:, t:t + 1], cache)
    for _ in range(n_new):
        logit = logits[0, -1].astype(np.float64)
        if temperature <= 0:
            nxt = int(np.argmax(logit))
        else:
            logit = logit / temperature
            if top_k and top_k > 0:
                if top_k < logit.shape[0]:
                    kth = np.sort(logit)[-top_k]
                    logit[logit < kth] = -1e30
            p = softmax(logit, axis=-1)
            nxt = int(rng.choice(len(p), p=p))
        out.append(nxt)
        logits = forward_step(params, cfg, np.asarray([[nxt]], dtype=np.int64), cache)
    return np.asarray(out, dtype=np.int64)
