"""混合超级引擎 v2.0 测试套件。

覆盖：
  1. Tier-0 精确求解器（正确性 / 优先级 / 大数 / 失败回退）
  2. Tier-1 极速路由决策
  3. Tier-2 云端客户端（本地 Mock 服务器端到端 + 重试 + 失败）
  4. 混合链路降级（云端挂掉时 fallback 到本地）
  5. 本地微模型意图打分（有检查点时）
"""

import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wcm import router
from wcm import solver
from wcm.cloud import CloudError, available, chat

PASS, FAIL = 0, 0

CKPT = os.path.join("artifacts", "wcm_micro_int8.npz")
HAS_CKPT = os.path.exists(CKPT)


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}  {detail}")


# ---------------- Tier-0 求解器 ----------------

def test_solver():
    print("== Tier-0 精确求解器 ==")
    cases = [
        ("10 + 5", "15"),
        ("10+5", "15"),
        ("10 + 5 = ?", "15"),
        ("2 + 3 * 4", "14"),                    # 优先级
        ("(2 + 3) * 4", "20"),                  # 括号
        ("2 ^ 10", "1024"),                     # 兼容 ^ 乘方
        ("10 / 4", "2.5"),                      # 分数精确
        ("999999999 * 999999999", "999999998000000001"),  # 大数
        ("100 - 37", "63"),
    ]
    for text, want in cases:
        res = solver.solve_text(text)
        check(f"solver({text!r}) == {want}",
              res["ok"] and res["value_str"] == want,
              f"got {res}")

    res = solver.solve_text("10 / 0")
    check("solver(10/0) 失败回退", not res["ok"] and res["expr"] is not None,
          f"got {res}")

    res = solver.solve_text("你好，今天天气怎么样")
    check("solver(非算术) 不抽取", not res["ok"] and res["expr"] is None)

    res = solver.solve_text("")
    check("solver(空串) 安全", not res["ok"] and res["expr"] is None)


# ---------------- Tier-1 路由 ----------------

def test_router():
    print("== Tier-1 极速路由 ==")
    intent, info = router.decide("10 + 5 = ?")
    check("calc 意图 + 精确求解", intent == "calc" and info["solved"]
          and info["value_str"] == "15", f"got {intent} {info}")
    intent, info = router.decide("100 - 37")
    check("calc 减法", intent == "calc" and info["value_str"] == "63")

    intent, info = router.decide("帮我写一个 Python 快速排序函数")
    check("code 意图", intent == "code", f"got {intent}")

    intent, info = router.decide("分析一下 Transformer 注意力机制")
    check("reason 意图", intent == "reason", f"got {intent}")

    intent, info = router.decide("10 / 0 等于几")
    check("不可解算术 → calc 未解出（交云端）",
          intent == "calc" and not info["solved"], f"got {intent} {info}")


# ---------------- Tier-2 云端（Mock 服务器） ----------------

class MockHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        server = self.server
        if getattr(server, "fail_next", 0) > 0:
            server.fail_next -= 1
            payload = b'{"error": "simulated 500"}'
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        body = json.loads(self.rfile.read(0) or b"{}") if False else None
        payload = json.dumps({
            "id": "mock",
            "choices": [{"index": 0, "message": {"role": "assistant",
                                                  "content": "MOCK_CLOUD_ANSWER: 42"}}],
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


def test_cloud():
    print("== Tier-2 云端客户端（本地 Mock 端到端）==")
    server = ThreadingHTTPServer(("127.0.0.1", 0), MockHandler)
    server.fail_next = 0
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]

    saved = {k: os.environ.get(k) for k in
             ("DEEPSEEK_API_KEY", "WCM_API_KEY", "WCM_BASE_URL", "WCM_MODEL")}
    os.environ["WCM_API_KEY"] = "test-key"
    os.environ["WCM_BASE_URL"] = f"http://127.0.0.1:{port}/v1"
    os.environ["WCM_MODEL"] = "mock-model"
    try:
        check("available() 检测到 Key", available())
        out = chat([{"role": "user", "content": "1+1"}], retries=0)
        check("Mock 云端返回", out == "MOCK_CLOUD_ANSWER: 42", f"got {out!r}")

        server.fail_next = 1   # 第一次失败，重试一次后成功
        out = chat([{"role": "user", "content": "1+1"}], retries=1)
        check("失败 1 次后重试成功", out == "MOCK_CLOUD_ANSWER: 42", f"got {out!r}")

        server.fail_next = 5   # 持续失败 → CloudError
        try:
            chat([{"role": "user", "content": "1+1"}], retries=1)
            check("持续失败抛 CloudError", False, "未抛异常")
        except CloudError:
            check("持续失败抛 CloudError", True)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        server.shutdown()
        server.server_close()


# ---------------- 混合降级 ----------------

def test_fallback():
    print("== 混合链路降级 ==")
    from chat import answer
    # 未配置 Key：算术仍走 Tier-0（保证绝对正确）
    out, meta = answer("10 + 5", mode="auto")
    check("无 Key 时 10+5 == 15", "15" in out and meta["route"].startswith("Tier-0"),
          f"got {out!r} {meta}")
    # 指向不存在的端点 → 云端失败 → 本地微模型兜底（有检查点时）
    if HAS_CKPT:
        os.environ["WCM_API_KEY"] = "test-key"
        os.environ["WCM_BASE_URL"] = "http://127.0.0.1:1/v1"
        try:
            out, meta = answer("你好，介绍一下你自己", mode="auto")
            check("云端挂掉降级本地微模型",
                  meta["route"].startswith("Tier-1") and "降级" in meta["note"],
                  f"got {meta}")
        finally:
            os.environ.pop("WCM_API_KEY", None)
            os.environ.pop("WCM_BASE_URL", None)
    else:
        check("本地检查点存在（降级测试跳过）", False,
              "artifacts/wcm_micro_int8.npz 缺失")


# ---------------- Tier-1.5 本地模型意图打分 ----------------

def test_llm_scorer():
    print("== 本地微模型意图打分（第二意见；规则层才是主路由）==")
    if not HAS_CKPT:
        check("本地检查点存在（跳过）", False, "artifacts/wcm_micro_int8.npz 缺失")
        return
    scorer = router.LLMScorer(CKPT)
    s_calc = scorer.nll("7 * 8 = ", "7 * 8 = ")
    s_code = scorer.nll("7 * 8 = ", "Python: ")
    check("算术在 calc 前缀下 NLL 更低", s_calc < s_code,
          f"calc={s_calc:.3f} code={s_code:.3f}")
    intent, info = scorer.intent("Python: print(")
    finite = all(isinstance(v, float) and v == v and abs(v) != float("inf")
                 for v in info["llm_scores"].values())
    check("三类意图均产生有限分数", finite, f"got {info['llm_scores']}")
    # 自然语言意图分类对 4.55M 合成语料模型是弱项 —— 主路由由规则层保证正确性
    intent, info = router.decide("帮我写一个 Python 快速排序函数", use_llm=True)
    check("混合路由：规则层主决策（code）", intent == "code"
          and isinstance(info.get("llm"), dict), f"got {intent}")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    test_solver()
    test_router()
    test_cloud()
    test_fallback()
    test_llm_scorer()
    print(f"\n{'=' * 48}")
    if FAIL == 0:
        print(f"全部通过（{PASS} 项检查）")
        sys.exit(0)
    print(f"失败 {FAIL} 项，通过 {PASS} 项")
    sys.exit(1)
