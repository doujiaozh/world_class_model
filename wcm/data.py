"""语料加载、tokenize、批窗口采样。"""
from __future__ import annotations

import numpy as np

from .tokenizer import BPETokenizer


def load_ids(text_path: str, tok: BPETokenizer) -> np.ndarray:
    with open(text_path, encoding="utf-8") as f:
        text = f.read()
    return tok.encode(text)


def batch_stream(ids: np.ndarray, batch: int, seq_len: int, rng):
    """无限随机采样 (x, y) 窗口：x = ids[t:t+L], y = ids[t+1:t+L+1]。"""
    n = ids.shape[0]
    assert n > seq_len + 1, "语料太短"
    while True:
        starts = rng.integers(0, n - seq_len - 1, size=batch)
        x = np.stack([ids[s:s + seq_len] for s in starts]).astype(np.int64)
        y = np.stack([ids[s + 1:s + seq_len + 1] for s in starts]).astype(np.int64)
        yield x, y


def make_windows(ids: np.ndarray, seq_len: int) -> np.ndarray:
    """把 token 序列切成不重叠的完整窗口，用于评测。"""
    n = (ids.shape[0] - 1) // seq_len
    return np.stack([ids[i * seq_len:(i + 1) * seq_len + 1] for i in range(n)]).astype(np.int64)
