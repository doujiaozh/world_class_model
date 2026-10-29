"""字节级 BPE 分词器（纯 NumPy/标准库，无外部依赖）。

- 训练：先按正则把语料切成"词"（字母串 / 数字串 / 单个其他字符），再在词内做
  频率驱动的贪心合并；词表 = 256 个字节 + (vocab_size - 256) 条合并规则。
- 编码：对每个词用"合并优先级表"反复合并当前词内秩最小的相邻对（GPT-2 式）。
- 解码：逐 token 还原字节串，经 latin-1 读回字符串再拼回原文。
"""
from __future__ import annotations

import json
import re
from collections import Counter

import numpy as np

# 预分词正则：字母串 / 数字串 / 空格串 / 换行 / 其他单字符 / 兜底空白。
# 每个字符恰好落入一个分支，保证 encode->decode 完全可逆。
WORD_RE = re.compile(r"[A-Za-z]+|[0-9]+|[ \t]+|[\r\n]+|[^\sA-Za-z0-9]|\s")


class BPETokenizer:
    def __init__(self):
        self.merges: list[tuple[int, int]] = []      # 合并规则（按学习顺序）
        self._rank: dict[tuple[int, int], int] = {}  # (left, right) -> 合并优先级
        self._vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self.special_ids: dict[str, int] = {}

    # ---------------- 训练 ----------------
    def train(self, text: str, vocab_size: int, verbose: bool = False):
        assert vocab_size >= 256, "vocab_size 至少为 256（字节基础词表）"
        words = Counter(w for w in WORD_RE.findall(text))
        # 每个唯一词 -> 符号序列（初始为 utf-8 字节 id 列表）及出现次数
        sym_words = [list(w.encode("utf-8")) for w in words]
        counts = [words[w] for w in words]
        order = list(range(len(sym_words)))
        # 用索引表加速：pair -> 出现该 pair 的词下标集合
        pair_counts: Counter = Counter()
        word_pairs: dict[int, Counter] = {}
        for wi, syms in enumerate(sym_words):
            pc = Counter(zip(syms, syms[1:]))
            word_pairs[wi] = pc
            for pair, n in pc.items():
                pair_counts[pair] += n * counts[wi]

        merges = []
        next_id = 256
        vocab = {i: bytes([i]) for i in range(256)}
        while next_id < vocab_size:
            if not pair_counts:
                break
            # 取频率最高（并列时取字典序最小，保证可复现）
            best = max(pair_counts.items(), key=lambda kv: (kv[1], -sum(kv[0])))[0]
            if pair_counts[best] < 1:
                break
            merges.append(best)
            vocab[next_id] = vocab[best[0]] + vocab[best[1]]
            # 在所有含该 pair 的词里执行合并
            for wi in list(order):
                syms = sym_words[wi]
                pc = word_pairs[wi]
                if best not in pc:
                    continue
                out = []
                i = 0
                n = len(syms)
                while i < n:
                    if i < n - 1 and (syms[i], syms[i + 1]) == best:
                        out.append(next_id)
                        i += 2
                    else:
                        out.append(syms[i])
                        i += 1
                # 更新该词的 pair 计数
                new_pc = Counter(zip(out, out[1:]))
                for pair, n_old in pc.items():
                    delta = new_pc.get(pair, 0) - n_old
                    if delta:
                        pair_counts[pair] += delta * counts[wi]
                        if pair_counts[pair] <= 0:
                            del pair_counts[pair]
                word_pairs[wi] = new_pc
                sym_words[wi] = out
            pair_counts.pop(best, None)  # 已彻底合并
            next_id += 1

        self.merges = merges
        self._rank = {m: i for i, m in enumerate(merges)}
        self._vocab = vocab
        if verbose:
            print(f"BPE 训练完成：词表 {next_id}（{len(merges)} 条合并）", flush=True)
        return self

    # ---------------- 编码 ----------------
    def _merge(self, syms: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
        out = []
        i = 0
        n = len(syms)
        while i < n:
            if i < n - 1 and (syms[i], syms[i + 1]) == pair:
                out.append(new_id)
                i += 2
            else:
                out.append(syms[i])
                i += 1
        return out

    def encode(self, text: str) -> np.ndarray:
        ids = []
        cache: dict[str, list[int]] = {}
        for w in WORD_RE.findall(text):
            enc = cache.get(w)
            if enc is None:
                enc = self._encode_word_uncached(w)
                cache[w] = enc
            ids.extend(enc)
        return np.asarray(ids, dtype=np.int64)

    def _encode_word_uncached(self, word: str) -> list[int]:
        syms = list(word.encode("utf-8"))
        # 反查表：pair -> new id（用 merges 顺序构造）
        if len(syms) < 2 or not self._rank:
            return syms
        while True:
            best, best_rank = None, None
            for pair in zip(syms, syms[1:]):
                r = self._rank.get(pair)
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank, best = r, pair
            if best is None:
                return syms
            syms = self._merge(syms, best, 256 + best_rank)

    # ---------------- 解码 ----------------
    def decode(self, ids) -> str:
        out = bytearray()
        for i in ids:
            i = int(i)
            if i in self.special_ids.values():
                continue
            out.extend(self._vocab[i])
        return out.decode("utf-8", errors="replace")

    # ---------------- 持久化 ----------------
    def save(self, path: str):
        obj = {"merges": [list(m) for m in self.merges]}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f)

    def load(self, path: str):
        with open(path, encoding="utf-8") as f:
            obj = json.load(f)
        self.merges = [tuple(m) for m in obj["merges"]]
        self._rank = {m: i for i, m in enumerate(self.merges)}
        self._vocab = {i: bytes([i]) for i in range(256)}
        for i, (a, b) in enumerate(self.merges):
            self._vocab[256 + i] = self._vocab[a] + self._vocab[b]
        return self

    @property
    def vocab_size(self) -> int:
        return 256 + len(self.merges)
