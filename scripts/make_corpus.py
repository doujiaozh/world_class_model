"""生成纯本地的"课程式"语料 data/corpus.txt（可复现，无任何网络依赖）。

类别覆盖通用能力的原语：
  - 加减乘除算术
  - 乘法表
  - 计数与等差序列
  - 三段论逻辑推理
  - Python 代码小片段（含正确输出）
  - 常识事实
  - 字符串反转
  - 数值比较
每行一条样本，BPE 会自然学到跨类别的共享子词。
"""
from __future__ import annotations

import argparse
import os
import random

OPERATORS = ["+", "-", "x"]


def gen_arithmetic(rng):
    a = rng.randint(0, 99)
    b = rng.randint(0, 99)
    op = rng.choice(OPERATORS)
    if op == "+":
        r = a + b
    elif op == "-":
        if a < b:
            a, b = b, a
        r = a - b
    else:
        a, b = rng.randint(0, 12), rng.randint(0, 12)
        r = a * b
    return f"{a} {op} {b} = {r}"


def gen_table(rng):
    a = rng.randint(2, 12)
    b = rng.randint(2, 12)
    return f"{a} x {b} = {a * b}"


def gen_counting(rng):
    start = rng.randint(1, 40)
    step = rng.choice([1, 2, 5])
    n = 6
    return " ".join(str(start + i * step) for i in range(n))


def SYLLOGISMS():
    return [
        ("cats", "animals", "mammals"), ("dogs", "animals", "carnivores"),
        ("roses", "flowers", "plants"), ("gold", "metals", "elements"),
        ("sharks", "fish", "predators"), ("eagles", "birds", "hunters"),
        ("apples", "fruits", "foods"), ("cars", "vehicles", "machines"),
        ("doctors", "humans", "workers"), ("triangles", "shapes", "polygons"),
    ]


def gen_syllogism(rng):
    a, b, c = rng.choice(SYLLOGISMS())
    kind = rng.random()
    if kind < 0.5:
        return (f"all {a} are {b} . all {b} are {c} . "
                f"therefore all {a} are {c} .")
    x = rng.choice(["tigers", "lions", "tables", "rivers", "stones", "birds"])
    if kind < 0.75:
        return (f"all {a} are {b} . {x} is not a {b} . "
                f"therefore {x} is not a {a} .")
    return (f"all {a} are {b} . {x} is a {a} . "
            f"therefore {x} is a {b} .")


CODE_TPL = [
    ("add", "def add ( a , b ) :\n    return a + b"),
    ("sub", "def sub ( a , b ) :\n    return a - b"),
    ("max", "def my_max ( a , b ) :\n    if a > b :\n        return a\n    return b"),
    ("even", "def is_even ( n ) :\n    return n % 2 == 0"),
    ("len", "def my_len ( s ) :\n    count = 0\n    for c in s :\n        count += 1\n    return count"),
]


def gen_code(rng):
    name, body = rng.choice(CODE_TPL)
    a = rng.randint(0, 20)
    b = rng.randint(0, 20)
    if name == "add":
        out = a + b
        call = f"add ( {a} , {b} )"
    elif name == "sub":
        out = abs(a - b) if rng.random() < 0.5 else a - b
        call = f"sub ( {a} , {b} )"
    elif name == "max":
        out = max(a, b)
        call = f"my_max ( {a} , {b} )"
    elif name == "even":
        out = "true" if a % 2 == 0 else "false"
        call = f"is_even ( {a} )"
    else:
        return body  # len 模板无固定输出
    return f"{body}\nprint ( {call} )\n{out}"


FACTS = [
    "the sun rises in the east",
    "water boils at 100 degrees celsius",
    "a year has 365 days",
    "humans need water to live",
    "fire is hot",
    "ice is cold",
    "the sky is blue",
    "birds can fly",
    "fish live in water",
    "the earth goes around the sun",
    "two plus two equals four",
    "a week has seven days",
    "gravity pulls things down",
    "trees need light to grow",
    "sound travels through air",
]


def gen_fact(rng):
    f = rng.choice(FACTS)
    if rng.random() < 0.5:
        return f"fact : {f}"
    return f"did you know {f} ?"


def gen_reverse(rng):
    n = rng.randint(3, 8)
    s = "".join(rng.choice("abcdef") for _ in range(n))
    return f"reverse ( {s} ) = {s[::-1]}"


def gen_compare(rng):
    a, b = rng.randint(0, 99), rng.randint(0, 99)
    if rng.random() < 0.5:
        bigger = max(a, b)
        return f"which is bigger {a} or {b} ? {bigger}"
    smaller = min(a, b)
    return f"which is smaller {a} or {b} ? {smaller}"


GENS = [
    (gen_arithmetic, 0.30), (gen_table, 0.08), (gen_counting, 0.06),
    (gen_syllogism, 0.12), (gen_code, 0.12), (gen_fact, 0.14),
    (gen_reverse, 0.08), (gen_compare, 0.10),
]


def gen_line(rng):
    u = rng.random()
    acc = 0.0
    for fn, w in GENS:
        acc += w
        if u <= acc:
            return fn(rng)
    return gen_fact(rng)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lines", type=int, default=120000)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "data", "corpus.txt"))
    args = ap.parse_args()
    out = os.path.normpath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    rng = random.Random(args.seed)
    with open(out, "w", encoding="utf-8") as f:
        for i in range(args.lines):
            f.write(gen_line(rng) + "\n")
    size = os.path.getsize(out)
    print(f"生成 {args.lines} 行 -> {out} ({size/1024:.1f} KiB)", flush=True)


if __name__ == "__main__":
    main()
