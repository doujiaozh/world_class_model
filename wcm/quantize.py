"""int8 对称量化（按行/输出通道），把 fp32 权重压缩到 1/4。

checkpoint 格式：每个 2D 权重 w 变成 (int8 矩阵, fp32 行缩放向量)；
1D 的 RMSNorm 权重保持 fp32。加载时反量化回 fp32 参与（推理）前向。
"""
from __future__ import annotations

import os

import numpy as np

from .config import Config
from .utils import load_checkpoint


def quantize_params(params: dict) -> dict:
    out = {}
    for k, v in params.items():
        if v.ndim == 2:
            scale = np.max(np.abs(v), axis=1, keepdims=True) / 127.0
            scale = np.maximum(scale, 1e-8)
            q = np.round(v / scale).clip(-127, 127).astype(np.int8)
            out[k] = q
            out[k + "__scale"] = scale.astype(np.float32).ravel()
        else:
            out[k] = v
    return out


def dequantize_params(qparams: dict) -> dict:
    out = {}
    for k, v in qparams.items():
        if k.endswith("__scale"):
            continue
        if v.ndim == 2:
            out[k] = (v.astype(np.float32) * qparams[k + "__scale"][:, None])
        else:
            out[k] = v
    return out


def save_quantized(path: str, params: dict, cfg: Config, meta: dict | None = None):
    q = quantize_params(params)
    arrays = dict(q)
    arrays["__config__"] = np.frombuffer(cfg.to_json().encode("utf-8"), dtype=np.uint8)
    arrays["__meta__kind"] = np.frombuffer(b"int8-quantized", dtype=np.uint8)
    if meta:
        for k, v in meta.items():
            arrays[f"__meta__{k}"] = np.frombuffer(str(v).encode("utf-8"), dtype=np.uint8)
    np.savez(path, **arrays)
    return os.path.getsize(path)


def load_quantized(path: str):
    """返回 (反量化后的 fp32 params, Config, meta)。"""
    z = np.load(path, allow_pickle=False)
    qparams, meta = {}, {}
    cfg = None
    for k in z.files:
        if k == "__config__":
            cfg = Config.from_json(bytes(z[k].astype(np.uint8)).decode("utf-8"))
        elif k.startswith("__meta__"):
            meta[k[len("__meta__"):]] = bytes(z[k].astype(np.uint8)).decode("utf-8")
        else:
            qparams[k] = z[k]
    if cfg is None:
        raise ValueError(f"量化检查点 {path} 缺少 __config__")
    return dequantize_params(qparams), cfg, meta
