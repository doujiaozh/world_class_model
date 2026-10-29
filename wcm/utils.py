"""通用工具：随机数、softmax、计时、检查点读写。"""
from __future__ import annotations

import os
import time

import numpy as np

from .config import Config


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    m = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / np.sum(e, axis=axis, keepdims=True)


class Timer:
    def __init__(self, label: str):
        self.label = label

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.dt = time.perf_counter() - self.t0
        print(f"[timer] {self.label}: {self.dt:.2f}s", flush=True)


def save_checkpoint(path: str, params: dict, cfg: Config, extra: dict | None = None):
    """把参数与配置序列化进一个 .npz。"""
    arrays = dict(params)
    arrays["__config__"] = np.frombuffer(cfg.to_json().encode("utf-8"), dtype=np.uint8)
    if extra:
        for k, v in extra.items():
            arrays[f"__meta__{k}"] = np.frombuffer(str(v).encode("utf-8"), dtype=np.uint8)
    np.savez(path, **arrays)


def load_checkpoint(path: str):
    """返回 (params dict, Config, meta dict)。"""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    z = np.load(path, allow_pickle=False)
    params, meta = {}, {}
    cfg = None
    for k in z.files:
        if k == "__config__":
            cfg = Config.from_json(bytes(z[k].astype(np.uint8)).decode("utf-8"))
        elif k.startswith("__meta__"):
            meta[k[len("__meta__"):]] = bytes(z[k].astype(np.uint8)).decode("utf-8")
        else:
            params[k] = z[k]
    if cfg is None:
        raise ValueError(f"检查点 {path} 缺少 __config__")
    return params, cfg, meta
