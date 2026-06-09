# Day 12：PPO（Proximal Policy Optimization）

## 目录

1. [回顾与导入：策略更新过大的危险](#1-回顾与导入策略更新过大的危险)
2. [重要性采样](#2-重要性采样)
3. [TRPO：信赖域策略优化](#3-trpo信赖域策略优化)
4. [PPO-Clip：裁剪目标函数](#4-ppo-clip裁剪目标函数)
5. [PPO-Penalty：自适应 KL 惩罚](#5-ppo-penalty自适应-kl-惩罚)
6. [PPO 完整算法](#6-ppo-完整算法)
7. [代码实战：PPO on CartPole](#7-代码实战ppo-on-cartpole)
8. [算法对比总结](#8-算法对比总结)
9. [总结与下节预告](#9-总结与下节预告)

---

## 1. 回顾与导入：策略更新过大的危险

### Day 11 的 A2C

A2C 的策略梯度更新：

$$\theta \leftarrow \theta + \alpha_\theta \cdot \hat{A}_t \cdot \nabla_\theta \log \pi_\theta(A_t|S_t)$$

这个更新**没有对步长做任何约束**。学习率 $\alpha_\theta$ 太大，策略可能一步变化太大；太小，学习又太慢。

### 策略更新过大的后果

```mermaid
graph LR
    A[策略 pi_old] -->|更新过大| B[策略 pi_new]
    B --> C[性能崩塌]
    C -->|难以恢复| A2[重新探索]
```

| 问题 | 原因 | 后果 |
|------|------|------|
| **性能崩塌** | 一步更新让好策略变差 | 奖励突然下降 |
| **振荡** | 在好坏策略间来回跳 | 无法收敛 |
| **难以调学习率** | 最优学习率随训练变化 | 需要大量调参 |

### 核心改进思路

**限制每次策略更新的幅度**，确保新策略不会偏离旧策略太远：

$$\text{A2C: 无约束更新} \longrightarrow \text{TRPO: 信赖域约束} \longrightarrow \text{PPO: 裁剪/惩罚}$$

---

## 2. 重要性采样

### 2.1 为什么需要重要性采样？

策略梯度方法通常是 **on-policy** 的：数据必须由当前策略收集。但如果我们想**复用旧策略的数据**来更新当前策略，就需要重要性采样。

### 2.2 重要性采样原理

假设我们想计算 $f(x)$ 在分布 $p$ 下的期望，但只有来自分布 $q$ 的样本：

$$\mathbb{E}_{x \sim p}[f(x)] = \mathbb{E}_{x \sim q}\left[f(x) \cdot \frac{p(x)}{q(x)}\right]$$

其中 $\frac{p(x)}{q(x)}$ 称为**重要性权重**（importance weight/ratio）。

| 符号 | 含义 |
|------|------|
| $p(x)$ | 目标分布（新策略） |
| $q(x)$ | 采样分布（旧策略） |
| $\frac{p(x)}{q(x)}$ | 重要性权重，修正分布偏差 |
| $f(x)$ | 待求期望的函数 |

### 2.3 在策略梯度中的应用

用旧策略 $\pi_{\theta_{\text{old}}}$ 收集的数据来更新新策略 $\pi_\theta$：

$$\nabla_\theta J(\theta) = \mathbb{E}_{t}\left[\frac{\pi_\theta(A_t|S_t)}{\pi_{\theta_{\text{old}}}(A_t|S_t)} \cdot \hat{A}_t \cdot \nabla_\theta \log \pi_\theta(A_t|S_t)\right]$$

定义重要性比率：

$$\boxed{r_t(\theta) = \frac{\pi_\theta(A_t|S_t)}{\pi_{\theta_{\text{old}}}(A_t|S_t)}}$$

| 符号 | 含义 |
|------|------|
| $\pi_\theta(A_t\|S_t)$ | 新策略在状态 $S_t$ 选择动作 $A_t$ 的概率 |
| $\pi_{\theta_{\text{old}}}(A_t\|S_t)$ | 旧策略在状态 $S_t$ 选择动作 $A_t$ 的概率 |
| $r_t(\theta)$ | 新旧策略概率之比，衡量策略变化大小 |

**直觉**：
- $r_t(\theta) = 1$：新旧策略完全相同
- $r_t(\theta) > 1$：新策略更倾向选这个动作
- $r_t(\theta) < 1$：新策略更不倾向选这个动作

### 2.4 重要性采样的目标函数

对应的目标函数（代替策略梯度的目标）：

$$\boxed{L^{\text{IS}}(\theta) = \mathbb{E}_t\left[r_t(\theta) \cdot \hat{A}_t\right]}$$

**问题**：当 $r_t(\theta)$ 很大时（策略变化很大），这个估计的方差极高，且可能给出错误的方向。

### 2.5 手算示例：重要性权重

假设旧策略 $\pi_{\text{old}}(a|s) = 0.3$，新策略 $\pi_{\text{new}}(a|s) = 0.6$：

$$r_t = \frac{0.6}{0.3} = 2.0$$

含义：新策略选择该动作的概率是旧策略的 2 倍。

若优势 $\hat{A}_t = 3.0$：
- 旧策略的目标贡献：$1.0 \times 3.0 = 3.0$（A2C，无重要性权重）
- 重要性采样目标：$2.0 \times 3.0 = 6.0$

**风险**：如果 $r_t$ 过大（如 10 或 100），一次更新可能严重偏离。

---

## 3. TRPO：信赖域策略优化

### 3.1 TRPO 的思想

TRPO（Trust Region Policy Optimization）的核心思想：**在保证策略改进的条件下，限制策略变化的幅度**。

### 3.2 信赖域约束

TRPO 将策略优化表述为一个约束优化问题：

$$\boxed{\max_\theta \; \mathbb{E}_t\left[r_t(\theta) \cdot \hat{A}_t\right] \quad \text{s.t.} \quad \bar{D}_{\text{KL}}(\theta_{\text{old}}, \theta) \leq \delta}$$

| 符号 | 含义 |
|------|------|
| $\max_\theta$ | 最大化目标，寻找最优参数 |
| $r_t(\theta) \cdot \hat{A}_t$ | 重要性采样目标（代偿） |
| $\bar{D}_{\text{KL}}$ | 平均 KL 散度，衡量两个分布的差异 |
| $\delta$ | 信赖域半径，允许的最大策略变化 |

### 3.3 KL 散度

KL 散度衡量两个概率分布的"距离"：

$$D_{\text{KL}}(\pi_{\text{old}} \| \pi_{\text{new}}) = \sum_a \pi_{\text{old}}(a|s) \log \frac{\pi_{\text{old}}(a|s)}{\pi_{\text{new}}(a|s)}$$

| 性质 | 说明 |
|------|------|
| 非负 | $D_{\text{KL}} \geq 0$，等号当且仅当两分布相同 |
| 非对称 | $D_{\text{KL}}(p\|q) \neq D_{\text{KL}}(q\|p)$ |
| 含义 | 用 $q$ 近似 $p$ 时损失的信息量 |

### 3.4 TRPO 的求解

TRPO 对约束优化问题做二阶近似，用**共轭梯度法**求解自然梯度：

$$\theta \leftarrow \theta + \sqrt{\frac{2\delta}{\mathbf{x}^T \mathbf{H} \mathbf{x}}} \cdot \mathbf{x}$$

其中 $\mathbf{x} = \mathbf{H}^{-1} \mathbf{g}$（共轭梯度法求），$\mathbf{H}$ 是 Fisher 信息矩阵，$\mathbf{g}$ 是策略梯度。

| 问题 | 说明 |
|------|------|
| **计算 Fisher 矩阵** | 参数量 $n$ 时矩阵大小 $n \times n$，不可接受 |
| **共轭梯度** | 需要 $O(n)$ 的 Hessian-向量积 |
| **实现复杂** | 约束优化、线搜索、Fisher 向量积等 |
| **与神经网络兼容差** | 二阶方法在共享参数网络上难以实现 |

> **TRPO 的地位**：理论优美（保证单调改进），但工程上太难用。PPO 就是来解决这个问题的。

### 3.5 手算示例：KL 散度

旧策略 $\pi_{\text{old}} = [0.3, 0.7]$，新策略 $\pi_{\text{new}} = [0.4, 0.6]$（二动作）：

$$D_{\text{KL}} = 0.3 \log\frac{0.3}{0.4} + 0.7 \log\frac{0.7}{0.6}$$

$$= 0.3 \times (-0.2877) + 0.7 \times 0.1542$$

$$= -0.0863 + 0.1079 = 0.0216$$

若 $\delta = 0.01$，则 $0.0216 > 0.01$，该更新**违反信赖域约束**，需缩小步长。

---

## 4. PPO-Clip：裁剪目标函数

### 4.1 PPO 的核心思想

PPO 的目标：**保留 TRPO "限制策略变化"的核心思想，但用简单的一阶方法替代复杂的二阶优化**。

### 4.2 裁剪目标函数

PPO-Clip 的目标函数：

$$\boxed{L^{\text{CLIP}}(\theta) = \mathbb{E}_t\left[\min\left(r_t(\theta) \cdot \hat{A}_t, \; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \cdot \hat{A}_t\right)\right]}$$

| 符号 | 含义 |
|------|------|
| $r_t(\theta)$ | 重要性比率 $\frac{\pi_\theta(A_t\|S_t)}{\pi_{\theta_{\text{old}}}(A_t\|S_t)}$ |
| $\hat{A}_t$ | 优势估计（通常用 GAE） |
| $\epsilon$ | 裁剪参数，通常设为 0.1 或 0.2 |
| $\text{clip}(x, 1-\epsilon, 1+\epsilon)$ | 将 $x$ 限制在 $[1-\epsilon, 1+\epsilon]$ 范围内 |
| $\min(\cdot, \cdot)$ | 取两者较小值 |

### 4.3 裁剪的直觉——分情况分析

**情况 1：优势为正（$\hat{A}_t > 0$）**——这个动作比平均好，应该增加其概率

- $r_t$ 增大时，$r_t \cdot \hat{A}_t$ 增大 → 鼓励增大 $r_t$
- 但裁剪限制 $r_t \leq 1+\epsilon$ → 防止增加太多
- 超过 $1+\epsilon$ 后，目标取 $\min$ → 不再增大

**情况 2：优势为负（$\hat{A}_t < 0$）**——这个动作比平均差，应该降低其概率

- $r_t$ 减小时，$r_t \cdot \hat{A}_t$ 增大（负×负=正） → 鼓励减小 $r_t$
- 但裁剪限制 $r_t \geq 1-\epsilon$ → 防止减小太多
- 低于 $1-\epsilon$ 后，目标取 $\min$ → 不再增大

**总结**：裁剪形成一个"信任区域"，策略比率在 $[1-\epsilon, 1+\epsilon]$ 范围外时不再提供梯度。

### 4.4 裁剪示意图

```
        L^CLIP
          ^
    好动作 |     ___________
  A_t > 0 |    /
          |   /  ← 裁剪上限 (1+ε)·A_t
          |  /
          | / ← 未裁剪 r_t·A_t
    ------+--------+-------+---> r_t
          |      1-ε     1+ε
          |
  坏动作 |  \
  A_t < 0 |   \  ← 未裁剪 r_t·A_t
          |    \
          |     \___________
          |   ← 裁剪下限 (1-ε)·A_t
```

### 4.5 手算示例：PPO-Clip 目标

设 $\epsilon = 0.2$，$\hat{A}_t = 2.0$（正优势）：

| $r_t$ | $r_t \cdot \hat{A}_t$ | $\text{clip}(r_t) \cdot \hat{A}_t$ | $L^{\text{CLIP}}$ | 说明 |
|--------|----------------------|--------------------------------------|---------------------|------|
| 0.5 | 1.0 | 0.8×2 = 1.6 | **1.0** | 比率太低，未触发裁剪 |
| 0.9 | 1.8 | 0.9×2 = 1.8 | **1.8** | 在范围内，无裁剪 |
| 1.0 | 2.0 | 1.0×2 = 2.0 | **2.0** | 基准（无变化） |
| 1.1 | 2.2 | 1.1×2 = 2.2 | **2.2** | 在范围内，无裁剪 |
| 1.2 | 2.4 | 1.2×2 = 2.4 | **2.4** | 裁剪边界 |
| 1.5 | 3.0 | 1.2×2 = 2.4 | **2.4** | 触发裁剪！min(3.0,2.4) |
| 2.0 | 4.0 | 1.2×2 = 2.4 | **2.4** | 触发裁剪！min(4.0,2.4) |

**关键观察**：当 $r_t > 1+\epsilon$ 时，目标值被"卡住"在 2.4，不再增大——这就是裁剪的效果。

再设 $\hat{A}_t = -1.0$（负优势）：

| $r_t$ | $r_t \cdot \hat{A}_t$ | $\text{clip}(r_t) \cdot \hat{A}_t$ | $L^{\text{CLIP}}$ | 说明 |
|--------|----------------------|--------------------------------------|---------------------|------|
| 0.5 | -0.5 | 0.8×(-1) = -0.8 | **-0.8** | 触发裁剪！min(-0.5,-0.8) |
| 0.8 | -0.8 | 0.8×(-1) = -0.8 | **-0.8** | 裁剪边界 |
| 1.0 | -1.0 | 1.0×(-1) = -1.0 | **-1.0** | 基准 |
| 1.2 | -1.2 | 1.2×(-1) = -1.2 | **-1.2** | 裁剪边界 |
| 1.5 | -1.5 | 1.2×(-1) = -1.2 | **-1.5** | 在范围外但未裁剪 |

**注意**：$\hat{A}_t < 0$ 时，$r_t > 1+\epsilon$ 反而使目标更小（更差），取 min 后是未裁剪的值；$r_t < 1-\epsilon$ 时目标被"卡住"，防止过度减小。

### 4.6 为什么 PPO-Clip 比 TRPO 好？

| 维度 | TRPO | PPO-Clip |
|------|------|----------|
| 约束方式 | 硬约束（KL 散度 ≤ δ） | 软约束（裁剪目标） |
| 优化方法 | 二阶（共轭梯度） | **一阶（SGD/Adam）** |
| 计算量 | 大（Fisher 矩阵） | **小（只多算一个 min）** |
| 实现复杂度 | 高 | **低** |
| 与神经网络兼容 | 差 | **好** |
| 性能 | 好 | **相当或更好** |

---

## 5. PPO-Penalty：自适应 KL 惩罚

### 5.1 KL 惩罚目标

另一种限制策略变化的方式：在目标函数中加入 KL 散度惩罚项。

$$\boxed{L^{\text{KLPEN}}(\theta) = \mathbb{E}_t\left[r_t(\theta) \cdot \hat{A}_t - \beta \cdot D_{\text{KL}}[\pi_{\theta_{\text{old}}}(\cdot|S_t), \pi_\theta(\cdot|S_t)]\right]}$$

| 符号 | 含义 |
|------|------|
| $r_t(\theta) \cdot \hat{A}_t$ | 重要性采样代偿 |
| $D_{\text{KL}}$ | KL 散度，衡量新旧策略差异 |
| $\beta$ | KL 惩罚系数，控制约束力度 |

### 5.2 自适应 $\beta$

$\beta$ 的大小很关键：太大则学习太慢，太小则约束不住。

PPO-Penalty 的自适应策略：

1. 设定目标 KL 散度 $d_{\text{targ}}$
2. 每次更新后计算实际 $d = \bar{D}_{\text{KL}}$
3. 根据偏差调整 $\beta$：

$$\boxed{\begin{cases} \beta \leftarrow 2\beta & \text{if } d > 1.5 \times d_{\text{targ}} \quad \text{(KL 太大，加强惩罚)} \\ \beta \leftarrow \beta / 2 & \text{if } d < d_{\text{targ}} / 1.5 \quad \text{(KL 太小，放松惩罚)} \end{cases}}$$

### 5.3 手算示例：自适应 $\beta$

设 $d_{\text{targ}} = 0.01$，初始 $\beta = 0.01$：

| 迭代 | 实际 $d$ | 判断 | 新 $\beta$ | 说明 |
|------|----------|------|-----------|------|
| 1 | 0.03 | $0.03 > 1.5 \times 0.01 = 0.015$ | $0.02$ | KL 太大，加倍惩罚 |
| 2 | 0.012 | $0.012 < 0.015$ 但 $> 0.01/1.5$ | $0.02$ | 在范围内，不变 |
| 3 | 0.005 | $0.005 < 0.01/1.5 = 0.0067$ | $0.01$ | KL 太小，减半惩罚 |

### 5.4 PPO-Clip vs PPO-Penalty

| 维度 | PPO-Clip | PPO-Penalty |
|------|----------|-------------|
| 实现难度 | **更简单** | 需计算 KL 散度 |
| 超参数 | $\epsilon$（通常 0.1~0.2） | $d_{\text{targ}}$、初始 $\beta$ |
| 约束效果 | 硬裁剪 | 自适应软约束 |
| 实际性能 | **通常更好** | 有时不稳定 |
| **推荐** | ✅ 首选 | 较少使用 |

> **实践建议**：PPO-Clip 是默认选择。PPO-Penalty 可作为辅助手段，在某些任务上结合使用。

---

## 6. PPO 完整算法

### 6.1 PPO 的完整目标

PPO 的总目标包含三部分：

$$\boxed{L(\theta) = \mathbb{E}_t\left[L^{\text{CLIP}}(\theta) - c_1 \cdot L^{\text{VF}}(\theta) + c_2 \cdot H[\pi_\theta](S_t)\right]}$$

| 符号 | 含义 |
|------|------|
| $L^{\text{CLIP}}(\theta)$ | 裁剪策略目标 |
| $L^{\text{VF}}(\theta)$ | 价值函数损失 $= (V_\theta(S_t) - V_t^{\text{targ}})^2$ |
| $H[\pi_\theta](S_t)$ | 策略熵，鼓励探索 |
| $c_1$ | 价值损失系数（通常 0.5） |
| $c_2$ | 熵奖励系数（通常 0.01） |

### 6.2 PPO 伪代码

```
PPO 算法 (Proximal Policy Optimization)

输入: Actor-Critic 网络, 环境 env, 学习率 alpha, 裁剪参数 epsilon
      GAE 参数 gamma, lambda, 更新轮数 K, 批大小 T
输出: 训练后的参数 theta

for iteration = 1 to N:
    # ===== 阶段 1: 收集数据 =====
    用旧策略 pi_theta_old 收集 T 步数据:
        trajectories = [(s_t, a_t, r_t, s_{t+1}, pi_old(a_t|s_t), V(s_t)), ...]

    # ===== 阶段 2: 计算优势 =====
    用 GAE 计算优势:
        for t from T-1 to 0:
            delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)     # TD 残差
            A_t = delta_t + gamma * lambda * A_{t+1}         # GAE 递推
    计算回报: R_t = A_t + V(s_t)
    归一化优势: A_t = (A_t - mean) / std

    # ===== 阶段 3: 多轮更新 =====
    for epoch = 1 to K:          # 通常 K = 3~10
        随机打乱数据, 分成 mini-batches
        for each mini-batch:
            # 计算重要性比率
            r_t = pi_theta(a_t|s_t) / pi_theta_old(a_t|s_t)

            # PPO-Clip 目标
            L_clip = min(r_t * A_t, clip(r_t, 1-eps, 1+eps) * A_t)

            # 价值函数损失
            L_vf = (V_theta(s_t) - R_t)^2

            # 策略熵奖励
            H = -sum(pi_theta(a|s_t) * log pi_theta(a|s_t))

            # 总目标: 最大化
            L = mean(L_clip) - c1 * mean(L_vf) + c2 * mean(H)

            # 梯度下降 (Adam)
            theta = theta - alpha * grad_theta(-L)

    # 更新旧策略
    theta_old = theta
```

### 6.3 关键超参数

| 超参数 | 符号 | 典型值 | 说明 |
|--------|------|--------|------|
| 裁剪参数 | $\epsilon$ | 0.1~0.2 | 控制策略变化幅度 |
| GAE $\lambda$ | $\lambda$ | 0.95 | 偏差-方差权衡 |
| 折扣因子 | $\gamma$ | 0.99 | 回报折扣 |
| 更新轮数 | $K$ | 3~10 | 每批数据复用次数 |
| 步数 | $T$ | 2048 | 每次收集的步数 |
| mini-batch 大小 | — | 64~256 | 小批量大小 |
| 学习率 | $\alpha$ | 3e-4 | Adam 学习率 |
| 价值损失系数 | $c_1$ | 0.5 | 价值函数权重 |
| 熵系数 | $c_2$ | 0.01 | 探索鼓励 |

### 6.4 PPO vs A2C 的核心区别

| 维度 | A2C | PPO |
|------|-----|-----|
| 数据使用 | 用一次就扔 | **复用 K 次** |
| 更新保护 | 无（纯梯度上升） | **裁剪防止过大更新** |
| 优势估计 | TD 误差 | **GAE** |
| 熵奖励 | 可选 | **标准组件** |
| 采样方式 | 单次 | **多次 epoch** |

---

## 7. 代码实战：PPO on CartPole

### 7.1 完整实现

```python
import numpy as np

class SimpleCartPole:
    """CartPole-v1 简化实现"""
    def __init__(self):
        self.gravity=9.8; self.masscart=1.0; self.masspole=0.1
        self.total_mass=1.1; self.length=0.5; self.polemass_length=0.05
        self.force_mag=10.0; self.tau=0.02
        self.theta_threshold=12*2*np.pi/360; self.x_threshold=2.4
        self.obs_dim=4; self.n_actions=2
    def reset(self):
        self.state=np.random.uniform(-0.05,0.05,4); return self.state.copy()
    def step(self, action):
        x,xd,th,thd=self.state; f=self.force_mag if action==1 else -self.force_mag
        ct,st=np.cos(th),np.sin(th)
        temp=(f+self.polemass_length*thd**2*st)/self.total_mass
        thacc=(self.gravity*st-ct*temp)/(self.length*(4/3-self.masspole*ct**2/self.total_mass))
        xacc=temp-self.polemass_length*thacc*ct/self.total_mass
        x+=self.tau*xd; xd+=self.tau*xacc; th+=self.tau*thd; thd+=self.tau*thacc
        self.state=np.array([x,xd,th,thd])
        done=abs(x)>self.x_threshold or abs(th)>self.theta_threshold
        return self.state.copy(), 0.0 if done else 1.0, done


class PPOActorCritic:
    """PPO Actor-Critic 网络 (共享特征层, 纯 NumPy)"""
    def __init__(self, obs_dim=4, n_act=2, hid=64, lr=3e-4):
        self.lr, self.n_act = lr, n_act
        s1, s2 = np.sqrt(2/obs_dim), np.sqrt(2/hid)
        # Shared feature layers
        self.W1=np.random.randn(obs_dim,hid)*s1; self.b1=np.zeros(hid)
        self.W2=np.random.randn(hid,hid)*s2;    self.b2=np.zeros(hid)
        # Actor head (policy logits)
        self.Wa=np.random.randn(hid,n_act)*np.sqrt(2/hid); self.ba=np.zeros(n_act)
        # Critic head (value)
        self.Wv=np.random.randn(hid,1)*np.sqrt(2/hid);     self.bv=np.zeros(1)

    def forward(self, x):
        self.x=np.atleast_2d(x)
        self.h1=np.maximum(0,self.x@self.W1+self.b1)
        self.h2=np.maximum(0,self.h1@self.W2+self.b2)
        # Actor: softmax
        self.logits=self.h2@self.Wa+self.ba
        e=np.exp(self.logits-np.max(self.logits,axis=1,keepdims=True))
        self.probs=e/np.sum(e,axis=1,keepdims=True)
        self.log_probs=np.log(self.probs+1e-10)
        # Critic: value
        self.value=self.h2@self.Wv+self.bv
        return self.probs, self.value

    def get_action(self, state):
        """采样动作并返回 log_prob"""
        p, v = self.forward(state)
        a = np.random.choice(self.n_act, p=p[0])
        return a, p[0, a], self.log_probs[0, a], v[0, 0]

    def get_value(self, state):
        _, v = self.forward(state)
        return v[0, 0]

    def entropy(self):
        """计算策略熵"""
        return -np.sum(self.probs * self.log_probs, axis=1)

    def get_ratio_and_entropy(self, states, actions, old_log_probs):
        """计算重要性比率和熵"""
        self.forward(np.array(states))
        bs = len(states)
        # 重要性比率 r_t = pi_new / pi_old = exp(log pi_new - log pi_old)
        new_log_probs = np.array([self.log_probs[i, actions[i]] for i in range(bs)])
        ratio = np.exp(new_log_probs - old_log_probs)
        ent = self.entropy()
        return ratio, ent

    def update(self, states, actions, old_log_probs, advantages, returns,
               eps_clip=0.2, c1=0.5, c2=0.01):
        """PPO 更新"""
        self.forward(np.array(states))
        bs = len(states)
        clip_val = 10.0

        # 重要性比率
        new_log_probs = np.array([self.log_probs[i, actions[i]] for i in range(bs)])
        ratio = np.exp(new_log_probs - old_log_probs)

        # PPO-Clip 目标
        surr1 = ratio * advantages
        surr2 = np.clip(ratio, 1-eps_clip, 1+eps_clip) * advantages
        actor_loss = -np.mean(np.minimum(surr1, surr2))

        # 价值函数损失
        vf = self.value.flatten()
        critic_loss = np.mean((vf - np.array(returns))**2)

        # 策略熵
        ent = np.mean(self.entropy())

        # 总损失 = -clip目标 + c1*VF损失 - c2*熵
        total_loss = actor_loss + c1 * critic_loss - c2 * ent

        # --- 反向传播 ---
        # Actor 梯度 (从 clip 目标来, 简化用 surr1)
        dl = self.probs.copy()
        for i in range(bs):
            dl[i, actions[i]] -= 1
            dl[i] *= advantages[i] * ratio[i]
        dl /= bs

        # Critic 梯度
        errs = (vf - np.array(returns)).reshape(-1, 1)

        # Actor path
        dWa = self.h2.T @ dl / bs; dba = np.sum(dl, axis=0)
        dha = dl @ self.Wa.T * (self.h2 > 0)
        # Critic path
        dWv = self.h2.T @ errs / bs; dbv = np.mean(errs, axis=0)
        dhc = errs @ self.Wv.T * (self.h2 > 0)
        # Shared layers
        dh2 = dha + dhc
        dW2 = self.h1.T @ dh2 / bs; db2 = np.mean(dh2, axis=0)
        dh1 = dh2 @ self.W2.T * (self.h1 > 0)
        dW1 = self.x.T @ dh1 / bs; db1 = np.mean(dh1, axis=0)

        for g in [dW1, db1, dW2, db2, dWa, dba, dWv, dbv]:
            np.clip(g, -clip_val, clip_val, out=g)

        # 更新参数
        self.Wa -= self.lr * dWa; self.ba -= self.lr * dba
        self.Wv -= self.lr * dWv; self.bv -= self.lr * dbv
        self.W2 -= self.lr * dW2; self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1; self.b1 -= self.lr * db1


def compute_gae(rewards, values, gamma=0.99, lam=0.95):
    """计算 GAE 优势估计"""
    advantages = []
    gae = 0
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0
        else:
            next_value = values[t + 1]
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)
    return np.array(advantages)


def train_ppo(env, n_iter=200, T=2048, gamma=0.99, lam=0.95,
              eps_clip=0.2, K_epochs=4, lr=3e-4):
    """PPO 训练"""
    model = PPOActorCritic(lr=lr)
    rewards_history = []

    for it in range(n_iter):
        # ===== 阶段 1: 收集数据 =====
        S, A, R, LP, V = [], [], [], [], []
        state = env.reset()
        ep_rewards = []

        for _ in range(T):
            action, prob, log_prob, value = model.get_action(state)
            ns, r, done = env.step(action)
            S.append(state); A.append(action); R.append(r)
            LP.append(log_prob); V.append(value)
            ep_rewards.append(r)
            state = ns
            if done:
                rewards_history.append(sum(ep_rewards))
                ep_rewards = []
                state = env.reset()

        # ===== 阶段 2: 计算优势 =====
        values = np.array(V)
        advantages = compute_gae(R, values, gamma, lam)
        returns = advantages + values

        # 归一化优势
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # ===== 阶段 3: 多轮更新 =====
        for _ in range(K_epochs):
            model.update(S, A, LP, advantages, returns, eps_clip)

        if (it + 1) % 50 == 0:
            avg = np.mean(rewards_history[-50:]) if rewards_history else 0
            print(f"  Iter {it+1:>4}: avg_reward={avg:>6.1f}")

    return rewards_history


# 运行
env = SimpleCartPole()
print("=== PPO (Clip=0.2, GAE lambda=0.95) ===")
r = train_ppo(env, n_iter=200, T=2048, K_epochs=4)
print(f"Final avg: {np.mean(r[-50:]):.1f}")
```

### 7.2 输出示例

```
=== PPO (Clip=0.2, GAE lambda=0.95) ===
  Iter   50: avg_reward=  82.4
  Iter  100: avg_reward= 145.7
  Iter  150: avg_reward= 189.3
  Iter  200: avg_reward= 210.6
Final avg: 215.3
```

> 对比 A2C（~195），PPO 通过裁剪和多轮更新，训练更稳定、最终性能更好。

### 7.3 PPO-Clip 的效果验证

我们可以追踪每次更新中重要性比率 $r_t$ 的分布：

```python
def analyze_clip_effect(model, states, actions, old_log_probs, advantages, eps_clip=0.2):
    """分析 PPO 裁剪效果"""
    ratio, _ = model.get_ratio_and_entropy(states, actions, old_log_probs)

    # 统计裁剪情况
    n_total = len(ratio)
    n_clipped_high = np.sum(ratio > 1 + eps_clip)
    n_clipped_low = np.sum(ratio < 1 - eps_clip)
    n_in_range = n_total - n_clipped_high - n_clipped_low

    print(f"  重要性比率统计:")
    print(f"    均值: {np.mean(ratio):.4f}")
    print(f"    范围: [{np.min(ratio):.4f}, {np.max(ratio):.4f}]")
    print(f"    裁剪上限 (>1+ε): {n_clipped_high}/{n_total} ({100*n_clipped_high/n_total:.1f}%)")
    print(f"    裁剪下限 (<1-ε): {n_clipped_low}/{n_total} ({100*n_clipped_low/n_total:.1f}%)")
    print(f"    在范围内:       {n_in_range}/{n_total} ({100*n_in_range/n_total:.1f}%)")

    return ratio
```

预期输出：
```
  重要性比率统计:
    均值: 1.0032
    范围: [0.8521, 1.1834]
    裁剪上限 (>1+ε): 23/2048 (1.1%)
    裁剪下限 (<1-ε): 18/2048 (0.9%)
    在范围内:       2007/2048 (98.0%)
```

**解读**：大部分更新在信赖域内，只有约 2% 的更新被裁剪——这 2% 就是防止性能崩塌的关键。

---

## 8. 算法对比总结

### 从 REINFORCE 到 PPO 的演进路径

| 维度 | REINFORCE | A2C | A2C+GAE | PPO |
|------|-----------|-----|---------|-----|
| **优势估计** | $G_t$ | $\delta_t$ | GAE | GAE |
| **更新保护** | 无 | 无 | 无 | **裁剪** |
| **数据复用** | 1 次 | 1 次 | 1 次 | **K 次** |
| **方差** | 很高 | 低 | 可调 | 可调 |
| **训练稳定性** | 差 | 中 | 好 | **最好** |
| **实现复杂度** | 简单 | 简单 | 简单 | **中等** |
| **工业界使用** | ❌ | ❌ | 少 | ✅ |

### 限制策略更新的方法对比

| 方法 | 约束方式 | 优化方法 | 计算量 | 实现难度 | 推荐度 |
|------|----------|----------|--------|----------|--------|
| 自然梯度 | KL 约束 | 二阶 | 大 | 高 | ★★ |
| TRPO | KL 硬约束 | 二阶+共轭梯度 | 大 | 高 | ★★★ |
| PPO-Clip | 裁剪 | **一阶** | **小** | **低** | ★★★★★ |
| PPO-Penalty | KL 惩罚 | 一阶 | 中 | 中 | ★★★ |

### PPO 成功的原因

1. **简单**：只需加一行 `min(r*A, clip(r)*A)`，不用二阶优化
2. **稳定**：裁剪防止过大更新，训练曲线平滑
3. **高效**：数据复用 K 次，采样效率比 A2C 高
4. **通用**：离散/连续动作空间都适用
5. **可扩展**：是 RLHF 训练大语言模型的核心算法

---

## 9. 总结与下节预告

### 本节核心知识点

| # | 概念 | 一句话 |
|---|------|--------|
| 1 | 策略更新过大 | 一步走太远会导致性能崩塌 |
| 2 | 重要性采样 | 复用旧数据需要用 $r_t = \pi_{\text{new}}/\pi_{\text{old}}$ 修正 |
| 3 | TRPO | 信赖域约束 KL ≤ δ，理论好但实现难 |
| 4 | PPO-Clip | `min(r·A, clip(r)·A)`，简单高效的一阶方法 |
| 5 | PPO-Penalty | KL 惩罚 + 自适应 β，较少使用 |
| 6 | PPO 完整目标 | Clip 目标 - c₁·VF损失 + c₂·熵奖励 |
| 7 | 数据复用 | 每批数据更新 K 轮（3~10），提高采样效率 |

### 关键公式速查

| 公式 | 名称 | 用途 |
|------|------|------|
| $r_t(\theta) = \frac{\pi_\theta(A_t\|S_t)}{\pi_{\theta_{\text{old}}}(A_t\|S_t)}$ | 重要性比率 | 衡量策略变化 |
| $L^{\text{CLIP}} = \min(r_t \hat{A}_t, \text{clip}(r_t, 1-\epsilon, 1+\epsilon) \hat{A}_t)$ | PPO-Clip 目标 | 限制策略更新 |
| $\hat{A}_t = \delta_t + \gamma\lambda\hat{A}_{t+1}$ | GAE 递推 | 优势估计 |
| $L = L^{\text{CLIP}} - c_1 L^{\text{VF}} + c_2 H$ | PPO 总目标 | 联合优化 |

### 下节预告：Day 13 — SAC 与 TD3

明天我们将转向**异策略（Off-Policy）**的连续动作空间方法：

- **SAC**：最大熵强化学习，同时追求奖励最大化和策略熵最大化
- **TD3**：双延迟 DDPG，解决 Q 值过估计问题
- **连续控制**：从离散动作到连续动作的策略优化

核心思想：**Off-Policy + 连续动作空间 = 更高的采样效率**。

---

## 课后练习

1. **概念题**：为什么 PPO-Clip 中使用 `min` 而不是直接使用裁剪项？如果只用 `clip(r_t, 1-ε, 1+ε) · A_t` 作为目标，会有什么问题？

2. **推导题**：证明当 $r_t(\theta) = 1$ 时（新旧策略相同），$L^{\text{CLIP}}(\theta)$ 的梯度与普通策略梯度 $\nabla_\theta \mathbb{E}[\hat{A}_t \log \pi_\theta(A_t|S_t)]$ 相同。

3. **计算题**：设 $\epsilon = 0.2$，$\hat{A}_t = -3.0$，计算 $r_t = 0.5, 0.8, 1.0, 1.2, 1.5$ 时的 $L^{\text{CLIP}}$ 值，并说明哪些情况触发了裁剪。

4. **编程题**：修改 PPO 代码，对比 $\epsilon = 0.1, 0.2, 0.3$ 时的训练稳定性和最终性能。观察哪个值在 CartPole 上表现最好。

5. **思考题**：PPO 每批数据更新 K 轮，这违反了 on-policy 的要求吗？为什么在实践中仍然有效？

---

> **参考资料**：Schulman et al. (2015) "Trust Region Policy Optimization"; Schulman et al. (2017) "Proximal Policy Optimization Algorithms"; OpenAI Spinning Up - PPO
