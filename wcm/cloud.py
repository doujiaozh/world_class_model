"""Tier-2 云端大脑 —— OpenAI 兼容协议接入（默认 DeepSeek 官方端点）。

零第三方依赖：urllib + json。统一错误类型、超时、指数退避重试、SSE 流式解析。
没配置 API Key 时 available() 返回 False，路由自动降级到本地引擎。
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

DEFAULT_BASE = "https://api.deepseek.com/v1"
DEFAULT_MODEL = "deepseek-chat"

SYSTEM_PROMPT = (
    "你是世界级的编程与推理助手。回答要求：先给出简短严谨的推理过程，"
    "再给出最终答案；写代码时给出完整、可直接运行的代码，并附一句 usage；"
    "算术题逐步列式计算；回答语言跟随用户。"
)


class CloudError(RuntimeError):
    """云端调用失败（网络 / 鉴权 / 协议 / 超时）。"""


def config() -> dict:
    return {
        "api_key": os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("WCM_API_KEY"),
        "base_url": os.environ.get("WCM_BASE_URL", DEFAULT_BASE).rstrip("/"),
        "model": os.environ.get("WCM_MODEL", DEFAULT_MODEL),
    }


def available() -> bool:
    return bool(config()["api_key"])


def preprocess(prompt: str) -> list[dict]:
    """极其健壮的 Prompt 预处理：角色设定 + 任务包裹 + 结果约束。"""
    user = (
        f"{prompt.strip()}\n\n"
        "（要求：推理简短严谨；最终答案单独放在【答案】行；代码必须完整可运行）"
    )
    return [{"role": "user", "content": user}]


def chat(messages: list[dict], *, temperature: float = 0.3, max_tokens: int = 2048,
         timeout: float = 60.0, retries: int = 2) -> str:
    """非流式对话，返回助手文本。失败重试 retries 次后抛 CloudError。"""
    cfg = config()
    if not cfg["api_key"]:
        raise CloudError("未配置 API Key（设置环境变量 DEEPSEEK_API_KEY 后启用云端大脑）")
    body = {
        "model": cfg["model"],
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}] + messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    url = cfg["base_url"] + "/chat/completions"
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(body).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": "Bearer " + cfg["api_key"],
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8")
            obj = json.loads(raw)
            return obj["choices"][0]["message"]["content"]
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError,
                ValueError, KeyError, OSError) as e:
            last = e
            if attempt < retries:
                time.sleep(0.6 * (attempt + 1))
    raise CloudError(f"云端请求失败（重试 {retries} 次后）: {last}")


def stream(messages: list[dict], *, temperature: float = 0.3, max_tokens: int = 2048,
           timeout: float = 60.0):
    """流式对话生成器，逐 chunk 吐出文本（SSE 解析，打字机效果）。"""
    cfg = config()
    if not cfg["api_key"]:
        raise CloudError("未配置 API Key")
    body = {
        "model": cfg["model"],
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}] + messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
    }
    req = urllib.request.Request(
        cfg["base_url"] + "/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + cfg["api_key"],
            "Accept": "text/event-stream",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for line in resp:
                line = line.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[len("data:"):].strip()
                if payload == "[DONE]":
                    break
                try:
                    delta = json.loads(payload)["choices"][0]["delta"].get("content")
                except (ValueError, KeyError, IndexError):
                    continue
                if delta:
                    yield delta
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as e:
        raise CloudError(f"云端流式请求失败: {e}")
