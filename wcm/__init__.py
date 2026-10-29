"""world_class_model (wcm) —— 纯 NumPy 从零实现的通用语言模型。

- config:    模型配置与预设
- tokenizer: 可训练的字节级 BPE 分词器
- model:     手写 Transformer（RoPE / RMSNorm / SwiGLU / GQA / KV cache）含手动反向传播
- optimizer: AdamW
- data:      语料加载与批处理
- quantize:  int8 对称量化导出（权重文件 < 100MB 的关键）
"""
from .config import Config, PRESETS, get_config, count_params
from .model import init_params, forward, backward, forward_step, generate
from .tokenizer import BPETokenizer

__all__ = [
    "Config", "PRESETS", "get_config", "count_params",
    "init_params", "forward", "backward", "forward_step", "generate",
    "BPETokenizer",
]
