"""Tier-0 精确算术引擎 —— 确定性求解，永不幻觉。

安全 AST 白名单求值（无 eval / 无 exec），Fraction 分数运算保证精确，
除零 / 溢出 / 不支持的表达式自动失败，交由上层路由降级处理。
"""

from __future__ import annotations

import ast
import re
from fractions import Fraction

# 用户输入里最常见的算术片段：至少两个操作数 + 一个运算符（支持一层括号）
_EXPR_RE = re.compile(
    r"\d+(?:\.\d+)?(?:\s*[-+*/%^]\s*(?:\(\s*-?\d+(?:\.\d+)?\s*\)|-?\d+(?:\.\d+)?))+"
)
# 再覆盖一层：整串被括号包裹，如 "(2 + 3) * 4"
_EXPR_PAREN_RE = re.compile(
    r"\(\s*-?\d+(?:\.\d+)?(?:\s*[-+*/%^]\s*-?\d+(?:\.\d+)?\s*)+\)(?:\s*[-+*/%^]\s*-?\d+(?:\.\d+)?)*"
)

# 兼容写法归一化：^ 表示乘方，× ÷ − 全角符号
_XLAT = {"^": "**", "×": "*", "÷": "/", "−": "-"}

_SYM = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/", ast.Mod: "%", ast.Pow: "^"}


def _fmt(q: Fraction) -> str:
    if q.denominator == 1:
        return str(q.numerator)
    return f"{float(q):.6g}"


def _normalize(expr: str) -> str:
    for k, v in _XLAT.items():
        expr = expr.replace(k, v)
    return expr


def try_extract(text: str) -> str | None:
    """从自然文本里抽出第一个算术表达式；没有则返回 None。"""
    m = _EXPR_PAREN_RE.search(text) or _EXPR_RE.search(text)
    if not m:
        return None
    return _normalize(m.group(0))


class _SafeEval(ast.NodeVisitor):
    """白名单求值：只允许 + - * / % **、括号、数字字面量。"""

    def __init__(self) -> None:
        self.steps: list[str] = []

    def visit_Expression(self, node: ast.Expression):
        return self.visit(node.body)

    def visit_Constant(self, node: ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError("仅支持数字字面量")
        return Fraction(node.value)

    def visit_UnaryOp(self, node: ast.UnaryOp):
        v = self.visit(node.operand)
        if isinstance(node.op, ast.USub):
            return -v
        if isinstance(node.op, ast.UAdd):
            return v
        raise ValueError("不支持的一元运算")

    def visit_BinOp(self, node: ast.BinOp):
        a = self.visit(node.left)
        b = self.visit(node.right)
        sym = _SYM.get(type(node.op))
        if sym is None:
            raise ValueError(f"不支持的运算符: {type(node.op).__name__}")
        if isinstance(node.op, ast.Add):
            r = a + b
        elif isinstance(node.op, ast.Sub):
            r = a - b
        elif isinstance(node.op, ast.Mult):
            r = a * b
        elif isinstance(node.op, ast.Div):
            if b == 0:
                raise ZeroDivisionError("除以零")
            r = a / b
        elif isinstance(node.op, ast.Mod):
            if b == 0:
                raise ZeroDivisionError("对零取模")
            r = a % b
        elif isinstance(node.op, ast.Pow):
            if b.denominator != 1 or not (-64 <= b <= 64):
                raise ValueError("指数必须是 |e|<=64 的整数")
            r = a ** int(b)
        if r.denominator != 1 and len(str(r.numerator)) > 4000:
            raise ValueError("结果过长")
        self.steps.append(f"{_fmt(a)} {sym} {_fmt(b)} = {_fmt(r)}")
        return r


def solve(expr: str) -> tuple[Fraction, list[str]]:
    """求值并返回 (精确结果, 分步记录)。失败抛 ValueError / ZeroDivisionError。"""
    tree = ast.parse(expr.strip(), mode="eval")
    ev = _SafeEval()
    value = ev.visit(tree)
    return value, ev.steps


def solve_text(text: str) -> dict:
    """对外统一入口：从文本抽取表达式并求解。

    返回 {ok, expr, value, value_str, steps}，ok=False 表示精确求解失败。
    """
    expr = try_extract(text)
    if expr is None:
        return {"ok": False, "expr": None, "value": None, "value_str": None, "steps": []}
    try:
        value, steps = solve(expr)
    except (ValueError, ZeroDivisionError, SyntaxError, RecursionError):
        return {"ok": False, "expr": expr, "value": None, "value_str": None, "steps": []}
    return {"ok": True, "expr": expr, "value": value, "value_str": _fmt(value), "steps": steps}
