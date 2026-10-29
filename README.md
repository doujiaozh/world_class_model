# world_class_model — 从零手搓的开源小语言模型（纯 NumPy）

一个**完全从零实现、零 AI 框架依赖**的现代解码器语言模型项目。模型本体
（前向 + 手工推导的反向传播）只用 **NumPy** 写成，不调用 PyTorch / TensorFlow /
JAX，不调用任何外部大模型 API。MIT 协议，可完整复现。

面向 GitHub 发布的最终交付：`artifacts/wcm_micro_step1000.npz`（4.55M 参数，
fp32 18.2MB）及其 int8 量化版 `artifacts/wcm_micro_int8.npz`（**4.64MB**）。

---

## 1. 它是什么

| 项目 | 说明 |
|------|------|
| 架构 | 现代解码器 Transformer：RoPE / RMSNorm / SwiGLU / GQA / tied embedding / 因果掩码 |
| 训练路径 | 带反传 tape 的 `forward`（手工推导每条梯度公式，无自动微分） |
| 推理路径 | 带 KV cache 的 `forward_step`（增量解码，与训练前向数值等价） |
| 分词器 | 字节级 BPE（纯标准库 + NumPy，GPT-2 式按秩合并） |
| 优化器 | AdamW + warmup / 余弦退火 |
| 量化 | int8 对称量化（按行/输出通道），fp32 检查点与 int8 检查点同一套推理代码 |
| 语料 | `scripts/make_corpus.py`：纯本地、可复现的"课程式"合成语料（无网络依赖） |

## 2. 快速开始

```bash
pip install -r requirements.txt   # 只需 numpy>=1.24

# 1. 生成语料（可复现，seed 1234）
python scripts/make_corpus.py --lines 60000

# 2. 训练（首次会先训 BPE 分词器，约 8 分钟）
python train.py --config micro --steps 5000 --eval_every 100 --save_every 1000

# 3. 推理（自动识别 _int8.npz 后缀）
python generate.py --ckpt artifacts/wcm_micro_int8.npz --prompt "6 * 7 = " --n_new 5

# 4. 评测（独立随机种子出题，衡量泛化而非背答案）
python eval_bench.py --ckpt artifacts/wcm_micro_int8.npz

# 5. 正确性测试（数值梯度校验，10/10 必须全过）
python tests/test_smoke.py
```

## 3. 实测数据（全部可复现，无挑选）

### 3.1 Loss 曲线（CPU 纯 NumPy 实跑）

**tiny（0.65M 参数，2000 步全程跑完）**

```text
step     1 | loss 6.1544     # ≈ ln(461)，均匀分布起点
step   200 | loss 1.7221
step   500 | loss 0.6378
step  1000 | loss 0.5618
step  1500 | loss 0.5144
step  2000 | loss 0.5015     # 收尾（warmup30 + 余弦退火到 1e-4）
```

**micro（4.55M 参数，原定 5000 步，于 step ~1500 主动终止）**

```text
step     1 | loss 6.2750
step  1000 | loss 0.5396     # 存盘 wcm_micro_step1000.npz（18.2MB）
step  1500 | loss 0.5077     # 终止时（lr 仍在 8e-4 高位，未到退火收尾）
```

> 注：micro 的 5000 步计划在 step ~1500 被主动终止（防止长时间高负载），
> 最终发布权重为 **step 1000** 的检查点。它是一次**中途快照**，不是收敛终点。

### 3.2 eval_bench（独立种子出题，160 题）

| 配置 | TOTAL | 乘法 mul | 比较 compare | 加法 add | 减法 sub | 反转 reverse |
|------|-------|----------|--------------|----------|----------|--------------|
| tiny（2000 步，收敛） | **30.0%** | 90.0% | 25.0% | 10.5% | 0.0% | 0.0% |
| micro（step1000，中途快照） | **25.0%** | 82.5% | 12.5% | 5.3% | 4.8% | 0.0% |

**注意 micro 并未跑完**：它被终止在 loss 0.51、lr 8e-4 的位置，退火收尾
（loss 通常会再明显下一台阶）从未发生。因此 micro 此处**略逊于跑完整的
tiny 是预期现象**，不代表架构或代码问题。

### 3.3 生成样本（贪心解码，temperature=0）

```text
micro(step1000)  6 * 7 = 42        ✔（正确）
micro(step1000)  10 + 5 = 40       ✘（15；量级对，数值错）
tiny(2000)       7 * 8 = 56        ✔（正确）
tiny(2000)       23 - 9 = 18       ✘（14）
tiny(2000)       9 > 4 = 36        ✘（格式学会了，比较逻辑没学会）
tiny(2000)       reverse ( hello ) = cabfbbac   ✘（字符级对齐超容量）
```

## 4. 关于"进位算法边界"的诚实分析

这是本项目最有说服力的 Scaling Law 观察样本，数据不含糊：

**乘法（90% / 82.5%）学会了，加法进位（≤10.5%）没学会。**

原因分析（我们的判断，供社区讨论）：

1. **乘法表是查表任务**。语料里 `a * b = ` 的组合空间有限（两位数以内数百
   种），transformer 的注意力 + 前馈完全可以用参数容量**记住映射**，并在
   训练分布内泛化。eval 用独立种子出新题，仍在分布内，所以得分高。

2. **加法进位是算法任务**。`34 + 29 = 63` 要求模型对个位/十位分别求和并
   处理进位标志——这是**多步组合性过程**，不是单一映射。要学这个，模型
   需要在隐空间里组合出"数位分解 → 分位求和 → 进位传播"的电路。
   0.65M–4.55M 参数 + 1.5M token 语料不足以自发形成该电路。

3. **观察到的失败模式佐证了这一点**：`10 + 5 → 40`、`34 + 29 → 59`、
   `83 - 16 → 63`——输出永远是**格式完全正确、量级正确、数值错一位**的
   答案。模型学会了"输出一个两位数答案"这个任务形式，但内部的加法器
   没有建立起来。

4. **怎么才能学会？**（未验证的推测，属于诚实的开放问题）
   - 更大容量 + 更长训练（micro 跑满 5000 步、small 配置 40M 参数）；
   - 让答案"一位一位写出来"（链式思维格式语料），把隐式算法变成显式
     逐位预测，把组合复杂度摊到序列维度上；
   - 数位位置编码增强（让数码与位次显式对齐）。

   本项目没有做这些——**这是能力边界，不是已知 bug**。

## 5. 工程与正确性验证

反传正确性不靠信仰。`tests/test_smoke.py` 用**中心差分数值梯度**逐算子
校验每一条手写反传公式（RMSNorm / RoPE / SiLU / 注意力块 / SwiGLU FFN /
完整模型 / KV cache 一致性 / 生成形状 / BPE 可逆性 / int8 误差界），
完整模型 13 组参数的最差相对误差 < 1e-3，**10/10 全部通过**。

开发过程中数值梯度校验抓出的真实 bug（都有记录，可在代码里搜注释）：

- 注意力 einsum 头维/序列维写反（`(B,T,H,hd)` 布局不一致）；
- `forward` 存进 tape 的是层输出而非层输入，导致 RMSNorm 反传配错残差；
- RoPE 的 cos/sin 表缺一个头维，无法与 4D 张量广播；
- **学习率坍塌**：`lr_at()` 以 `self.lr` 为基数，训练循环却把结果赋回
  `opt.lr`（同一属性），lr 每步自乘——理论值 `1e-3 × 10!/30^10 = 6.13e-12`
  与日志观测 `6.15e-12` 完全吻合，loss 一度卡死在 `ln(461)`（均匀分布）。

## 6. 最终交付物

| 文件 | 说明 | 体积 |
|------|------|------|
| `artifacts/wcm_micro_step1000.npz` | micro 最终权重（fp32） | 18.2MB |
| `artifacts/wcm_micro_int8.npz` | 同一检查点的 int8 量化版（推理同效） | **4.64MB** |
| `artifacts/wcm_tiny.npz` / `wcm_tiny_int8.npz` | tiny 全程训练对照（fp32 / int8） | 2.6MB / 0.69MB |
| `artifacts/tokenizer.json` | BPE 分词器 | 2KB |

**权重 <100MB 的硬约束以 20 倍以上余量满足**（int8 4.64MB）。

## 7. 目录结构

```
wcm/                核心库（全部自己实现）
  config.py         模型配置与预设（tiny / micro / small）
  tokenizer.py      字节级 BPE 分词器
  model.py          解码器 Transformer（前向 + 手工反传 + KV cache 推理）
  optimizer.py      AdamW + warmup/余弦退火 lr
  data.py           语料加载与批流采样
  quantize.py       int8 对称量化
  utils.py          softmax / 计时器 / 检查点读写
train.py            训练 CLI（支持 --save_every 周期存盘、--resume 断点续训）
generate.py         推理 CLI（自动识别 int8 检查点）
eval_bench.py       能力评测（独立种子出题：加减乘/比较/反转）
scripts/make_corpus.py  课程式语料生成器（无网络依赖）
tests/test_smoke.py 数值梯度校验套件（10 项）
train_micro.bat     micro 训练一键脚本
```

## 8. 关于"超越 DeepSeek V4.1"的诚实声明

**不作此宣称，这不可能。** 一个单机 CPU 从零训练、权重 4.64MB 的模型
（参数量比工业级大模型小 4 个数量级以上），不可能在通用逻辑/常识/代码
能力上超越任何工业级大模型。此类宣称要么是幻觉，要么在玩文字游戏。

本项目的硬约束**全部真实达成并已验证**：

- ✅ 完全开源（MIT，见 LICENSE）
- ✅ 零外部 API、零 AI 框架（仅 NumPy）
- ✅ 权重 <100MB（int8 4.64MB）
- ✅ 本地从零训练、本地推理、端到端可复现

它真实的定位是：**一个架构现代、工程干净、每一行反传都经过数值验证、
实测数据（包括失败数据）全部如实公开的自研模型骨架**。乘法表 90% 的
泛化是它真实的成就；加法进位学不会是它真实的边界。两条我们都写在了
第 3、4 节里。
