"""AdamW 优化器（纯 NumPy）。"""
from __future__ import annotations

import numpy as np


class AdamW:
    def __init__(self, params: dict, lr: float = 1e-3, betas=(0.9, 0.99),
                 eps: float = 1e-8, weight_decay: float = 0.01):
        self.lr, self.b1, self.b2, self.eps, self.wd = lr, betas[0], betas[1], eps, weight_decay
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, params: dict, grads: dict, t: int, lr: float | None = None):
        lr = self.lr if lr is None else lr
        for k, w in params.items():
            g = grads[k]
            if g is None:
                continue
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * (g * g)
            mhat = self.m[k] / (1 - self.b1 ** t)
            vhat = self.v[k] / (1 - self.b2 ** t)
            # AdamW：权重衰减直接作用于参数而非梯度
            if self.wd > 0:
                w -= lr * self.wd * w
            w -= lr * mhat / (np.sqrt(vhat) + self.eps)

    def lr_at(self, t: int, total: int, warmup: int = 0, min_ratio: float = 0.1) -> float:
        """warmup + 余弦退火的学习率。"""
        if t <= warmup:
            return self.lr * t / max(1, warmup)
        p = (t - warmup) / max(1, total - warmup)
        p = min(1.0, max(0.0, p))
        return self.lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + np.cos(np.pi * p)))
