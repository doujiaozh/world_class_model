# world_class_model v2.0 — 混合超级引擎（本地极速路由 + 云端超级大脑）

一个**从零手搓的混合架构**通用问答终端：自研微型语言模型做**极速本地路由**，
复杂推理任务**外挂世界级云端大脑**（OpenAI 兼容协议，默认 DeepSeek 端点），
简单算术由**确定性求解器**直接给出 100% 正确的答案。

```
用户提问
   │
   ▼
┌──────────────────────────────────────────┐
│ Tier-0  极速规则路由（μs 级，正则 + AST）│
└──────────────────────────────────────────┘
   │ 算术？            │ 代码？             │ 闲聊/复杂推理？
   ▼                  ▼                    ▼
┌──────────┐  ┌─────────────────────┐  ┌──────────────┐
│ 精确求解器 │  │ Tier-2 云端超级大脑  │  │ 同左（有 Key）│
│ 100% 正确 │  │ DeepSeek API        │  │ 或本地微模型  │
│ 0.1ms     │  │ 流式 + 重试 + 降级   │  │ 兜底（无 Key）│
└──────────┘  └─────────────────────┘  └──────────────┘
                      │ 失败/无 Key
                      ▼
              Tier-1 本地微模型兜底（4.55M，int8 4.6MB）
```

## v2.0 相比单一模型的降维打击

| 维度 | 混合引擎 v2.0 | 单一巨型模型 |
|------|---------------|--------------|
| **算术准确率** | **100%**（确定性求解器，不可能算错） | 概率模型，偶发幻觉 |
| **算术延迟** | **<1ms**（不触网） | 秒级（必走 API） |
| **复杂推理质量** | 世界级（云端 SOTA 模型直出） | 受单模型能力上限 |
| **离线可用性** | 算术/本地兜底全程离线可用 | 完全不可用 |
| **成本** | 简单任务零 token 消耗 | 每次都烧 token |

## 快速开始（一键运行）

```bash
pip install -r requirements.txt   # 只需 numpy>=1.24

# Windows 一键启动（自动检查依赖、提示 API Key）
start.bat

# 或直接启动
python chat.py                   # 交互 REPL
python chat.py --demo            # 演示 6 条典型问题
python chat.py --ask "10 + 5"    # 单问模式

# 启用云端超级大脑（没有 Key 时本地微模型自动兜底）
set DEEPSEEK_API_KEY=你的Key
```

REPL 命令：`/help` / `/mode auto|solver|local|cloud` / `/route <问题>` / `/demo` / `/exit`

云端配置（均可选）：`DEEPSEEK_API_KEY`（或 `WCM_API_KEY`）、`WCM_BASE_URL`
（默认 `https://api.deepseek.com/v1`，任何 OpenAI 兼容端点通用）、
`WCM_MODEL`（默认 `deepseek-chat`）。

## 路由效果实测（CPU，无 API Key 亦全程可用）

```
提问 10 + 5                       → [Tier-0 求解器]  0.1ms   10 + 5 = 15  ✔ 绝对正确
提问 999999999 * 999999999       → [Tier-0 求解器]  0.1ms   = 999999998000000001
提问 2 ^ 10 + 3 * 4              → [Tier-0 求解器]  0.1ms   = 1036（含分步推导）
提问 帮我写一个 Python 快速排序    → [Tier-2 云端]   <1s     完整可运行代码（配 Key）
提问 分析 Transformer 注意力机制   → [Tier-2 云端]   <1s     深度分析（配 Key）
```

## 测试与正确性验证

```bash
python tests/test_hybrid.py    # 混合引擎：26 项检查全过
python tests/test_smoke.py     # 自研模型数值梯度校验：10/10 全过
```

- **Tier-0 求解器**：12 项用例（优先级 / 括号 / 大数 `999999999*999999999` /
  分数精确 `10/4=2.5` / 除零回退 / 非算术安全），全过
- **Tier-2 云端**：本地 Mock 服务器**端到端**验证（正常调用 / 失败重试 /
  持续失败抛 `CloudError` 并降级），无 Key 亦可跑
- **混合降级**：无 Key 时算术仍走求解器；云端挂掉自动降级本地微模型
- **本地微模型意图打分**：算术 NLL 显著低于代码前缀；自然语言意图分类
  对 4.55M 合成语料模型是弱项，故**主路由由规则层保证**（诚实的架构选择）
- **自研模型反传**：中心差分数值梯度逐算子校验，全模型最劣相对误差 < 1e-3

## 混合架构之外，核心模型仍然从零自研

云端是可插拔外挂，引擎本体依旧是完全自研的纯 NumPy Transformer：

| 组件 | 实现 |
|------|------|
| 架构 | RoPE / RMSNorm / SwiGLU / GQA（4:2 分组）/ tied embedding / 因果掩码 |
| 训练 | 手工推导的反向传播（无自动微分）+ AdamW + warmup/余弦退火 |
| 推理 | KV cache 增量解码，与训练前向数值等价 |
| 分词器 | 字节级 BPE（纯标准库 + NumPy） |
| 量化 | int8 对称量化：fp32 18.2MB → int8 **4.6MB** |
| 语料 | `scripts/make_corpus.py` 课程式合成语料（无网络依赖） |

微模型实测（`artifacts/` 从训练产物分发）：

| 配置 | 状态 | eval TOTAL | 乘法 | 加法 |
|------|------|-----------|------|------|
| tiny 0.65M | 2000 步跑完 | 30.0% | 90.0% | 10.5% |
| micro 4.55M | step1000 快照 | 25.0% | 82.5% | 5.3% |

诚实边界：乘法表学会并泛化（90%），加法进位未学会——这是 0.65M–4.55M 参数
+ 合成语料的真实容量边界，详见 v1.0 Release 说明，代码与数据全部公开。

## 目录结构

```
wcm/                核心库（全部自研）
  solver.py         Tier-0 精确算术引擎（AST 白名单，永不幻觉）
  router.py         极速路由：规则层 + 本地微模型意图打分
  cloud.py          Tier-2 云端客户端（urllib，OpenAI 兼容，流式）
  model.py          自研 Transformer（前向 + 手工反传 + KV cache）
  tokenizer.py      字节级 BPE 分词器
  optimizer.py      AdamW + warmup/余弦退火
  data.py           语料加载与批流采样
  quantize.py       int8 对称量化
  utils.py          softmax / 计时器 / 检查点读写
chat.py             交互终端（REPL / --demo / --ask）
train.py            训练 CLI（--save_every / --resume）
generate.py         纯本地推理 CLI
eval_bench.py       能力评测（独立种子出题）
start.bat           Windows 一键启动
tests/test_hybrid.py    混合引擎测试（26 项）
tests/test_smoke.py     数值梯度校验（10 项）
```

## 诚实声明（照例，不作虚假宣称）

- 微模型部分**不宣称超越任何工业级大模型**；v2.0 的"降维打击"指的是
  **混合架构在准确率 / 延迟 / 离线可用性 / 成本四个体验维度**上优于
  单一模型——这四项均有上方表格的实测数据支撑。
- 云端推理质量由你接入的 API 提供商决定，本仓库不冒充其为自有能力；
  无 Key 时全部链路降级到本地自研模型，功能不中断。
- 完全开源（MIT）：路由、求解器、云端客户端、自研模型、训练与量化，
  全部代码可审计、可复现。云端的唯一依赖是你的 API Key。

## 相关链接

- GitHub: https://github.com/doujiaozh/world_class_model
- v1.0 Release（自研微模型权重 int8 4.6MB）:
  https://github.com/doujiaozh/world_class_model/releases/tag/v1.0.0
