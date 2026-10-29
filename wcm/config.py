"""模型配置：预设 + 参数量统计。"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass


@dataclass
class Config:
    name: str = "tiny"
    vocab_size: int = 1024          # BPE 词表大小
    d_model: int = 128              # 隐藏维度
    n_layers: int = 4               # Transformer 层数
    n_heads: int = 4                # 查询头数
    n_kv_heads: int = 2             # 键/值头数（GQA，n_heads 的约数）
    ffn_hidden: int = 256           # SwiGLU FFN 中间维度
    max_seq_len: int = 256          # 支持的最大上下文长度
    rope_base: float = 10000.0      # RoPE 基频
    rms_eps: float = 1e-5           # RMSNorm epsilon

    @property
    def head_dim(self) -> int:
        assert self.d_model % self.n_heads == 0, "d_model 必须整除 n_heads"
        return self.d_model // self.n_heads

    @property
    def d_kv(self) -> int:
        return self.head_dim * self.n_kv_heads

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @staticmethod
    def from_json(s: str) -> "Config":
        return Config(**json.loads(s))


# 预设：tiny 用于冒烟测试/教学；micro 是 fp32 即可 <100MB 的实用配置；
# small 是本项目的"旗舰"配置，int8 量化后约 41MB，远低于 100MB 上限。
PRESETS = {
    # ~0.86M 参数，fp32 约 3.4MB —— CPU 分钟级训练
    "tiny": Config(name="tiny", vocab_size=1024, d_model=128, n_layers=4,
                   n_heads=4, n_kv_heads=2, ffn_hidden=256, max_seq_len=256),
    # ~6.5M 参数，fp32 约 26MB
    "micro": Config(name="micro", vocab_size=4096, d_model=256, n_layers=6,
                    n_heads=4, n_kv_heads=2, ffn_hidden=704, max_seq_len=512),
    # ~40.7M 参数：fp32 约 163MB（超预算），int8 量化后 ~41MB（达标）
    "small": Config(name="small", vocab_size=16000, d_model=512, n_layers=12,
                    n_heads=8, n_kv_heads=4, ffn_hidden=1280, max_seq_len=1024),
}


def get_config(name: str) -> Config:
    if name not in PRESETS:
        raise KeyError(f"未知预设 {name!r}，可选: {sorted(PRESETS)}")
    return PRESETS[name]


def count_params(cfg: Config) -> int:
    """可训练参数量（权重共享：LM head 与词嵌入同一矩阵）。"""
    d, h, dkv = cfg.d_model, cfg.ffn_hidden, cfg.d_kv
    per_layer = (d                      # ln1
                 + d * d                # attn q
                 + d * dkv              # attn k
                 + d * dkv              # attn v
                 + d * d                # attn o
                 + d                    # ln2
                 + d * h                # ffn gate
                 + d * h                # ffn up
                 + h * d)               # ffn down
    return cfg.vocab_size * d + cfg.n_layers * per_layer + d  # emb + layers + ln_f
