"""极速混合路由 —— 决定一个问题该交给谁。

分层架构（与 README 对应）：
  Tier-0 规则层        正则 + AST 白名单，μs 级，100% 可解释
  Tier-1 本地模型评分  已训练的 wcm 微模型给意图打分（NLL 比较），ms 级
  Tier-2 路由决策      算术→求解器；复杂任务→云端（有 key）否则本地
"""

from __future__ import annotations

import os

from . import solver as solver_mod

CODE_HINTS = ("代码", "写一个", "写一段", "编写", "实现", "函数", "def ", "class ",
              "import ", "function ", "bug", "报错", "python", "java", "sql",
              "算法", "正则", "程序")
HARD_HINTS = ("设计", "架构", "优化", "重构", "分析", "总结", "比较", "为什么",
              "解释", "论文", "方案", "证明", "推导")
CHAT_HINTS = ("你好", "你是", "天气", "讲讲", "聊聊", "今天", "名字")

INTENTS = ("calc", "code", "reason", "chat")

DEFAULT_CKPT = os.path.join("artifacts", "wcm_micro_int8.npz")


def rule_intent(text: str) -> tuple[str, dict]:
    """规则层：返回 (intent, info)。intent ∈ calc / code / reason / chat。"""
    info: dict = {"solved": False, "expr": None, "value": None,
                  "value_str": None, "steps": [], "llm": None}
    res = solver_mod.solve_text(text)
    info["expr"] = res["expr"]
    if res["ok"]:
        info.update(solved=True, value=res["value"],
                    value_str=res["value_str"], steps=res["steps"])
        return "calc", info
    if res["expr"] is not None:
        # 抽到了算术但精确求解失败（如除零）——交给更强的大脑
        return "calc", info
    if any(h in text for h in CODE_HINTS):
        return "code", info
    if any(h in text for h in HARD_HINTS):
        return "reason", info
    return "chat", info


class LLMScorer:
    """Tier-1：用已训练的本地微模型做意图评分。

    对每个意图类别，把 prefix + prompt 一起喂给模型，比较 prompt 部分
    的平均负对数似然（NLL），取最小者作为意图（zero-shot 分类）。
    """

    def __init__(self, ckpt: str = DEFAULT_CKPT):
        import numpy as np
        from . import model as model_mod
        from .quantize import load_quantized
        from .tokenizer import BPETokenizer
        from .utils import load_checkpoint, softmax

        self._np = np
        if ckpt.endswith("_int8.npz"):
            params, cfg, _meta = load_quantized(ckpt)
        else:
            params, cfg, _meta = load_checkpoint(ckpt)
        self.params, self.cfg = params, cfg
        tok = BPETokenizer()
        tok.load(os.path.join(os.path.dirname(ckpt), "tokenizer.json"))
        self.tok = tok
        self._model = model_mod
        self._softmax = softmax

    def nll(self, text: str, prefix: str) -> float:
        np = self._np
        p_ids = self.tok.encode(prefix)
        t_ids = self.tok.encode(prefix + text)
        ids = np.array(t_ids[: self.cfg.max_seq_len], dtype=np.int64)
        logits, _loss, _tape = self._model.forward(self.params, self.cfg, ids[None, :], None)
        logits = logits[0]  # (T, V)
        logp = np.log(self._softmax(logits[:-1], axis=-1) + 1e-30)  # (T-1, V)
        pos = np.arange(len(ids) - 1)
        mask = pos >= max(len(p_ids) - 1, 0)
        if not mask.any():
            return float("inf")
        nlls = -logp[pos[mask], ids[1:][mask]]
        return float(nlls.mean())

    def intent(self, text: str, prefixes: dict[str, str] | None = None) -> tuple[str, dict]:
        prefixes = prefixes or {
            "calc": "7 * 8 = ",
            "code": "Python: ",
            "chat": "Did you know ",
        }
        scores = {k: self.nll(text, p) for k, p in prefixes.items()}
        best = min(scores, key=scores.get)
        return best, {"llm_scores": scores}


def decide(text: str, use_llm: bool = False, ckpt: str = DEFAULT_CKPT) -> tuple[str, dict]:
    """综合决策。规则层为主，use_llm=True 时本地模型作为第二意见写入 info。"""
    intent, info = rule_intent(text)
    if use_llm:
        try:
            scorer = LLMScorer(ckpt)
            llm_intent, llm_info = scorer.intent(text)
            info["llm"] = {"intent": llm_intent, "scores": llm_info["llm_scores"]}
        except Exception as e:  # noqa: BLE001 - 降级路径，不能阻断主链路
            info["llm"] = {"error": f"{type(e).__name__}: {e}"}
    return intent, info
