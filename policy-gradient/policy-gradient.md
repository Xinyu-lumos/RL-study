# Day 10：策略梯度（Policy Gradient）

## 目录

1. [回顾与导入：从 Value-Based 到 Policy-Based](#1-回顾与导入从-value-based-到-policy-based)
2. [策略参数化](#2-策略参数化)
3. [策略梯度定理](#3-策略梯度定理)
4. [REINFORCE 算法](#4-reinforce-算法)
5. [REINFORCE with Baseline](#5-reinforce-with-baseline)
6. [策略梯度的方差问题](#6-策略梯度的方差问题)
7. [代码实战：CartPole 策略梯度](#7-代码实战cartpole-策略梯度)
8. [算法对比总结](#8-算法对比总结)
9. [总结与下节预告](#9-总结与下节预告)

---

## 1. 回顾与导入：从 Value-Based 到 Policy-Based

### Day 8-9 的范式

DQN 系列算法的核心思路：**先学 $Q^*(s,a)$，再从 Q 值提取策略 $\pi(s) = \arg\max_a Q(s,a)$**。

这叫 **Value-Based** 方法——间接地通过价值函数来找策略。

### Value-Based 的局限

| 局限 | 说明 | 例子 |
|------|------|------|
| **连续动作空间** | $\arg\max_a Q(s,a)$ 在连续空间无法遍历 | 机器人关节角度 $\in \mathbb{R}^n$ |
| **随机策略** | Q 值只能给出确定性策略 | 石头剪刀布需要随机出拳 |
| **策略退化** | Q 值微小时 $\arg\max$ 不稳定 | 两个动作 Q 值几乎相等 |
| **探索不足** | $\varepsilon$-贪心是硬编码的，不优雅 | 需要手动调 $\varepsilon$ |

### Policy-Based 的核心思想

> **直接参数化策略 $\pi_\theta(a|s)$，通过梯度上升最大化期望回报。**

$$\underbrace{Q^*(s,a) \to \pi^*(s)}_{\text{Value-Based: 间接}} \quad \longrightarrow \quad \underbrace{\pi_\theta(a|s) \to \text{最大化 } J(\theta)}_{\text{Policy-Based: 直接}}$$

### 两种范式对比

| 维度 | Value-Based | Policy-Based |
|------|-------------|--------------|
| 学习对象 | $Q(s,a)$ 或 $V(s)$ | $\pi_\theta(a\|s)$ |
| 策略类型 | 确定性（$\arg\max$） | 可以是随机的 |
| 动作空间 | 离散 | 离散/连续均可 |
| 收敛性 | 可能振荡 | 保证收敛到局部最优 |
| 方差 | 低 | 高 |
| 采样效率 | 高（off-policy） | 低（通常 on-policy） |

---

## 2. 策略参数化

### 2.1 策略的直接表示

用一个参数化函数直接表示策略：

$$\pi_\theta(a|s) = P(a|s;\theta)$$

其中 $\theta$ 是可学习参数（神经网络的权重）。

### 2.2 离散动作空间：Softmax 策略

对于离散动作空间，最常用的参数化方式：

$$\boxed{\pi_\theta(a|s) = \frac{\exp(h_\theta(s,a))}{\sum_{a'} \exp(h_\theta(s,a'))}}$$

| 符号 | 含义 |
|------|------|
| $h_\theta(s,a)$ | 策略网络对状态 $s$、动作 $a$ 输出的偏好值（logit） |
| $\exp(h_\theta(s,a))$ | 将偏好值转为正值 |
| $\sum_{a'} \exp(h_\theta(s,a'))$ | 归一化常数，确保概率和为 1 |

**直觉**：偏好值越高的动作，被选中的概率越大，但不会是 100%（保持探索）。

### 2.3 连续动作空间：高斯策略

对于连续动作空间，常用高斯分布：

$$\boxed{\pi_\theta(a|s) = \frac{1}{\sigma_\theta(s)\sqrt{2\pi}} \exp\left(-\frac{(a - \mu_\theta(s))^2}{2\sigma_\theta(s)^2}\right)}$$

| 符号 | 含义 |
|------|------|
| $\mu_\theta(s)$ | 策略网络输出的均值（最可能采取的动作） |
| $\sigma_\theta(s)$ | 策略网络输出的标准差（控制探索程度） |
| $a$ | 连续动作（如关节角度） |

**直觉**：$\mu_\theta(s)$ 是"最佳动作"，$\sigma_\theta(s)$ 控制围绕最佳动作的探索幅度。

### 2.4 策略参数化的优势

1. **天然随机策略**：不需要 $\varepsilon$-贪心，策略本身有探索性
2. **连续动作**：直接输出 $\mu$ 和 $\sigma$，无需 $\arg\max$
3. **平滑的策略变化**：参数 $\theta$ 小幅变化 → 策略小幅变化（比 $\arg\max$ 稳定）

---

## 3. 策略梯度定理

### 3.1 目标函数

策略梯度的目标是找到最优参数 $\theta^*$，使策略的期望回报最大化：

$$\boxed{J(\theta) = \mathbb{E}_{\pi_\theta}\left[\sum_{t=0}^{\infty} \gamma^t R_{t+1}\right] = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)]}$$

| 符号 | 含义 |
|------|------|
| $J(\theta)$ | 目标函数：策略 $\pi_\theta$ 下的期望折扣回报 |
| $\tau$ | 一条轨迹 $\tau = (S_0, A_0, R_1, S_1, A_1, R_2, \ldots)$ |
| $R(\tau)$ | 轨迹 $\tau$ 的折扣累积回报 |
| $\mathbb{E}_{\tau \sim \pi_\theta}$ | 在策略 $\pi_\theta$ 下采样轨迹的期望 |

### 3.2 核心推导：策略梯度定理

**目标**：计算 $\nabla_\theta J(\theta)$，即目标函数对参数的梯度。

**推导过程**：

**第一步**：展开期望

$$J(\theta) = \sum_\tau P(\tau|\theta) R(\tau)$$

其中 $P(\tau|\theta)$ 是策略 $\pi_\theta$ 产生轨迹 $\tau$ 的概率。

**第二步**：对 $\theta$ 求导

$$\nabla_\theta J(\theta) = \sum_\tau \nabla_\theta P(\tau|\theta) R(\tau)$$

**第三步**：利用对数导数技巧 $\nabla_\theta P = P \cdot \nabla_\theta \log P$

$$\nabla_\theta J(\theta) = \sum_\tau P(\tau|\theta) \nabla_\theta \log P(\tau|\theta) R(\tau) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\nabla_\theta \log P(\tau|\theta) R(\tau)\right]$$

**第四步**：展开 $\log P(\tau|\theta)$

一条轨迹的概率：

$$P(\tau|\theta) = \mu(S_0) \prod_{t=0}^{T-1} \pi_\theta(A_t|S_t) P(S_{t+1}|S_t,A_t)$$

取对数：

$$\log P(\tau|\theta) = \log \mu(S_0) + \sum_{t=0}^{T-1} \log \pi_\theta(A_t|S_t) + \sum_{t=0}^{T-1} \log P(S_{t+1}|S_t,A_t)$$

**第五步**：对 $\theta$ 求导——只有 $\pi_\theta$ 依赖 $\theta$

$$\nabla_\theta \log P(\tau|\theta) = \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(A_t|S_t)$$

**关键洞察**：转移概率 $P(S_{t+1}|S_t,A_t)$ 不依赖 $\theta$，求导后消失！这意味着我们**不需要知道环境模型**。

### 3.3 策略梯度定理（最终形式）

$$\boxed{\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\left(\sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(A_t|S_t)\right) R(\tau)\right]}$$

**逐项解释**：

| 项 | 含义 |
|----|------|
| $\nabla_\theta \log \pi_\theta(A_t\|S_t)$ | 在 $S_t$ 选中 $A_t$ 的对数概率对 $\theta$ 的梯度 |
| $\sum_{t=0}^{T-1}$ | 轨迹中所有时间步的梯度之和 |
| $R(\tau)$ | 整条轨迹的回报，作为加权信号 |
| $\mathbb{E}_{\tau \sim \pi_\theta}$ | 在当前策略下采样取期望 |

**直觉理解**：

- 如果轨迹回报 $R(\tau)$ 为正 → 增大这条轨迹上所有动作的选中概率
- 如果轨迹回报 $R(\tau)$ 为负 → 减小这条轨迹上所有动作的选中概率
- 回报越大，概率调整幅度越大

### 3.4 更一般的形式

策略梯度定理有一个更一般的形式，用**因果性**只考虑未来回报：

$$\boxed{\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\left[\sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(A_t|S_t) \cdot G_t\right]}$$

其中 $G_t = \sum_{k=t}^{T-1} \gamma^{k-t} R_{k+1}$ 是从 $t$ 时刻开始的折扣回报。

**为什么这是正确的**：$t$ 时刻的动作不应该为 $t$ 时刻之前的奖励负责（因果性），所以用 $G_t$ 替代 $R(\tau)$。

---

## 4. REINFORCE 算法

### 4.1 从定理到算法

策略梯度定理给出的是**期望**，我们用蒙特卡洛采样来近似：

$$\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i=1}^{N} \sum_{t=0}^{T_i-1} \nabla_\theta \log \pi_\theta(A_t^{(i)}|S_t^{(i)}) \cdot G_t^{(i)}$$

梯度上升更新：

$$\boxed{\theta \leftarrow \theta + \alpha \nabla_\theta J(\theta)}$$

### 4.2 REINFORCE 伪代码

```
REINFORCE 算法

输入: 策略网络 pi_theta, 学习率 alpha, 回合数 N
输出: 训练后的参数 theta

for episode = 1 to N:
    # 1. 用当前策略生成一条完整轨迹
    s = env.reset()
    trajectory = []
    while not done:
        a = 从 pi_theta(.|s) 中采样
        s_next, r, done = env.step(a)
        trajectory.append((s, a, r))
        s = s_next
    
    # 2. 计算每一步的回报 G_t (从后往前)
    G = 0
    for t = T-1, T-2, ..., 0:
        G = r_{t+1} + gamma * G    # 折扣累积回报
        G_t = G
    
    # 3. 更新策略参数
    for t = 0, 1, ..., T-1:
        theta += alpha * grad_log_pi(A_t|S_t) * G_t
        # 即: theta += alpha * (d/d_theta) log pi(A_t|S_t) * G_t
```

### 4.3 手算示例：2 状态 2 动作

假设一个简单环境：1 个状态，2 个动作（左/右），一个回合只有 1 步。

策略参数：$\theta = [\theta_1, \theta_2]$（两个动作的偏好值）

$$\pi_\theta(\text{左}|s) = \frac{e^{\theta_1}}{e^{\theta_1} + e^{\theta_2}}, \quad \pi_\theta(\text{右}|s) = \frac{e^{\theta_2}}{e^{\theta_1} + e^{\theta_2}}$$

初始 $\theta_1 = 0, \theta_2 = 0$，所以 $\pi(\text{左}) = \pi(\text{右}) = 0.5$。

**采样**：Agent 选了"右"，得到奖励 $R = +3$。

**计算梯度**：

$$\nabla_{\theta_2} \log \pi_\theta(\text{右}|s) = \nabla_{\theta_2} \left[\theta_2 - \log(e^{\theta_1} + e^{\theta_2})\right] = 1 - \pi(\text{右}|s) = 1 - 0.5 = 0.5$$

$$\nabla_{\theta_1} \log \pi_\theta(\text{右}|s) = 0 - \pi(\text{左}|s) = -0.5$$

**更新**（$\alpha = 0.1$）：

$$\theta_2 \leftarrow 0 + 0.1 \times 0.5 \times 3 = 0.15$$

$$\theta_1 \leftarrow 0 + 0.1 \times (-0.5) \times 3 = -0.15$$

**结果**：$\theta_2 > \theta_1$，下次"右"被选中的概率更大（$e^{0.15} > e^{-0.15}$）。

**直觉**：选"右"得到了正奖励 → 增加"右"的概率，减少"左"的概率。

---

## 5. REINFORCE with Baseline

### 5.1 高方差问题

REINFORCE 的梯度估计：

$$\hat{g} = \sum_t \nabla_\theta \log \pi_\theta(A_t|S_t) \cdot G_t$$

**问题**：$G_t$ 的方差可能非常大。如果所有回报都是正的（如 CartPole 每步 +1），则：

- 好的动作被正回报增强 ✓
- 坏的动作**也被正回报增强** ✗

只是好的增强更多，但坏的动作也在增加概率——低效学习。

### 5.2 Baseline 的引入

核心想法：从回报中减去一个基线 $b(S_t)$，使得好的动作得到正信号，坏的动作得到负信号：

$$\boxed{\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\left[\sum_t \nabla_\theta \log \pi_\theta(A_t|S_t) \cdot (G_t - b(S_t))\right]}$$

**为什么减去 baseline 不引入偏差？**

$$\mathbb{E}_{\pi_\theta}\left[\nabla_\theta \log \pi_\theta(A_t|S_t) \cdot b(S_t)\right] = b(S_t) \mathbb{E}_{a \sim \pi_\theta}\left[\nabla_\theta \log \pi_\theta(a|S_t)\right] = b(S_t) \cdot 0 = 0$$

因为 $\sum_a \pi_\theta(a|s) = 1$，所以 $\nabla_\theta \sum_a \pi_\theta(a|s) = \nabla_\theta 1 = 0$，从而：

$$\sum_a \nabla_\theta \pi_\theta(a|s) = \sum_a \pi_\theta(a|s) \nabla_\theta \log \pi_\theta(a|s) = 0$$

**结论**：减去 baseline 不改变梯度的期望（无偏），但能**大幅降低方差**。

### 5.3 最优 Baseline

理论上，使方差最小的最优 baseline 为：

$$b^*(S_t) = \frac{\mathbb{E}_{\pi_\theta}\left[(\nabla_\theta \log \pi_\theta(A_t|S_t))^2 G_t\right]}{\mathbb{E}_{\pi_\theta}\left[(\nabla_\theta \log \pi_\theta(A_t|S_t))^2\right]}$$

实践中最常用的 baseline 是**状态价值函数 $V^\pi(S_t)$**：

$$\boxed{\nabla_\theta J(\theta) \approx \sum_t \nabla_\theta \log \pi_\theta(A_t|S_t) \cdot (G_t - \hat{V}(S_t))}$$

**直觉**：$G_t - V(S_t)$ 告诉我们"实际得到的比预期好多少"——这就是**优势函数** $A(S_t, A_t)$ 的估计。

### 5.4 REINFORCE with Baseline 伪代码

```
REINFORCE with Baseline 算法

输入: 策略网络 pi_theta, 价值网络 V_w, 学习率 alpha_policy, alpha_value
输出: 训练后的参数 theta, w

for episode = 1 to N:
    # 1. 用当前策略生成一条完整轨迹
    生成 trajectory = [(S_0, A_0, R_1), (S_1, A_1, R_2), ...]
    
    # 2. 计算每一步的回报 G_t
    for t = T-1, ..., 0:
        G_t = R_{t+1} + gamma * G_{t+1}
    
    # 3. 更新价值网络 (回归目标 G_t)
    for t = 0, ..., T-1:
        delta_t = G_t - V_w(S_t)                  # TD 误差
        w += alpha_value * delta_t * grad_w V_w(S_t)  # 最小化 (G_t - V)^2
    
    # 4. 更新策略网络 (用优势作为权重)
    for t = 0, ..., T-1:
        A_t = G_t - V_w(S_t)                      # 优势估计
        theta += alpha_policy * A_t * grad_theta log pi_theta(A_t|S_t)
```

### 5.5 无 Baseline vs 有 Baseline 对比

| 维度 | REINFORCE | REINFORCE + Baseline |
|------|-----------|---------------------|
| 梯度权重 | $G_t$ | $G_t - V(S_t)$ |
| 偏差 | 无偏 | 无偏 |
| 方差 | 高 | **显著降低** |
| 收敛速度 | 慢 | **更快** |
| 额外开销 | 无 | 需要训练价值网络 |
| 正负信号 | 全正（或全负） | 有正有负 |

---

## 6. 策略梯度的方差问题

### 6.1 方差来源

策略梯度估计的方差主要来自：

1. **轨迹的随机性**：同一策略产生不同轨迹，回报差异大
2. **信用分配**：整条轨迹的回报 $G_t$ 归因于每一步动作，但不是每步都同等重要
3. **时间相关性**：早期动作的梯度受后续所有奖励影响

### 6.2 降低方差的常用技巧

| 技巧 | 方法 | 原理 |
|------|------|------|
| **Baseline** | 减去 $V(S_t)$ | 让权重有正有负 |
| **优势估计** | 用 $A_t = G_t - V(S_t)$ | 只强化比预期好的动作 |
| **多条轨迹** | 增大 batch size | 平均降低方差 |
| **奖励归一化** | 标准化 $G_t$ | 稳定梯度量级 |

### 6.3 方差-偏差权衡的统一视角

```
高方差 ←─────────────────────────────────→ 低偏差

MC 回报 G_t              TD(0): R + gamma*V(S_{t+1})
(完整回报, 高方差)         (一步自举, 低方差但引入偏差)

       n-step return (n 越大越接近 MC)
```

REINFORCE 用 $G_t$（MC 回报），方差高但无偏。Day 11 的 Actor-Critic 将用 TD 误差替换 $G_t$，降低方差但引入少量偏差。

---

## 7. 代码实战：CartPole 策略梯度

### 7.1 环境介绍

CartPole：连续状态空间（4维），离散动作空间（2个），目标是保持杆子不倒。

### 7.2 完整实现

```python
import numpy as np

class SimpleCartPole:
    """CartPole-v1 简化实现"""
    def __init__(self):
        self.gravity = 9.8; self.masscart = 1.0; self.masspole = 0.1
        self.total_mass = 1.1; self.length = 0.5; self.polemass_length = 0.05
        self.force_mag = 10.0; self.tau = 0.02
        self.theta_threshold = 12 * 2 * np.pi / 360
        self.x_threshold = 2.4
        self.obs_dim = 4; self.n_actions = 2

    def reset(self):
        self.state = np.random.uniform(-0.05, 0.05, 4)
        return self.state.copy()

    def step(self, action):
        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag
        cos_theta, sin_theta = np.cos(theta), np.sin(theta)
        temp = (force + self.polemass_length * theta_dot**2 * sin_theta) / self.total_mass
        theta_acc = (self.gravity * sin_theta - cos_theta * temp) / \
                    (self.length * (4.0/3.0 - self.masspole * cos_theta**2 / self.total_mass))
        x_acc = temp - self.polemass_length * theta_acc * cos_theta / self.total_mass
        x += self.tau * x_dot; x_dot += self.tau * x_acc
        theta += self.tau * theta_dot; theta_dot += self.tau * theta_acc
        self.state = np.array([x, x_dot, theta, theta_dot])
        done = abs(x) > self.x_threshold or abs(theta) > self.theta_threshold
        return self.state.copy(), 0.0 if done else 1.0, done


class PolicyNetwork:
    """Softmax 策略网络 (纯 NumPy)"""
    def __init__(self, obs_dim=4, n_actions=2, hidden=64, lr=0.01):
        self.lr = lr; self.n_actions = n_actions
        s1, s2 = np.sqrt(2/obs_dim), np.sqrt(2/hidden)
        self.W1 = np.random.randn(obs_dim, hidden) * s1; self.b1 = np.zeros(hidden)
        self.W2 = np.random.randn(hidden, n_actions) * s2; self.b2 = np.zeros(n_actions)

    def forward(self, x):
        self.x = np.atleast_2d(x)
        self.h1 = np.maximum(0, self.x @ self.W1 + self.b1)
        self.logits = self.h1 @ self.W2 + self.b2
        # Softmax
        exp_l = np.exp(self.logits - np.max(self.logits, axis=1, keepdims=True))
        self.probs = exp_l / np.sum(exp_l, axis=1, keepdims=True)
        return self.probs

    def sample(self, state):
        probs = self.forward(state)[0]
        return np.random.choice(self.n_actions, p=probs)

    def update(self, states, actions, advantages):
        """REINFORCE 梯度更新"""
        bs = len(states)
        self.forward(np.array(states))
        # 计算梯度: d/d_theta log pi(a|s) * advantage
        dlogits = self.probs.copy()
        for i in range(bs):
            dlogits[i, actions[i]] -= 1  # grad of log-softmax = one_hot - probs
            dlogits[i] *= advantages[i]   # 乘以优势
        dlogits /= bs

        # 反向传播
        clip = 10.0
        dW2 = self.h1.T @ dlogits; db2 = np.sum(dlogits, axis=0)
        dh1 = dlogits @ self.W2.T * (self.h1 > 0)
        dW1 = self.x.T @ dh1; db1 = np.sum(dh1, axis=0)
        for g in [dW1, db1, dW2, db2]: np.clip(g, -clip, clip, out=g)
        self.W2 -= self.lr * dW2; self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1; self.b1 -= self.lr * db1


class ValueNetwork:
    """状态价值网络 V(s) (用作 baseline)"""
    def __init__(self, obs_dim=4, hidden=64, lr=0.01):
        self.lr = lr
        s1, s2 = np.sqrt(2/obs_dim), np.sqrt(2/hidden)
        self.W1 = np.random.randn(obs_dim, hidden) * s1; self.b1 = np.zeros(hidden)
        self.W2 = np.random.randn(hidden, 1) * s2;       self.b2 = np.zeros(1)

    def forward(self, x):
        self.x = np.atleast_2d(x)
        self.h1 = np.maximum(0, self.x @ self.W1 + self.b1)
        self.v = self.h1 @ self.W2 + self.b2
        return self.v

    def update(self, states, returns):
        bs = len(states)
        self.forward(np.array(states))
        errs = self.v.flatten() - np.array(returns)
        errs_2d = errs.reshape(-1, 1)
        dW2 = self.h1.T @ errs_2d / bs; db2 = np.mean(errs_2d, axis=0)
        dh1 = errs_2d @ self.W2.T * (self.h1 > 0)
        dW1 = self.x.T @ dh1 / bs; db1 = np.mean(dh1, axis=0)
        clip = 10.0
        for g in [dW1, db1, dW2, db2]: np.clip(g, -clip, clip, out=g)
        self.W2 -= self.lr * dW2; self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1; self.b1 -= self.lr * db1


def train_reinforce(env, n_episodes=500, gamma=0.99, policy_lr=0.003, value_lr=0.01,
                    use_baseline=True, verbose=True):
    policy = PolicyNetwork(lr=policy_lr)
    value = ValueNetwork(lr=value_lr) if use_baseline else None
    rewards_history = []

    for ep in range(n_episodes):
        # 1. 采样一条轨迹
        state = env.reset()
        ep_states, ep_actions, ep_rewards = [], [], []
        done = False
        while not done:
            action = policy.sample(state)
            next_state, reward, done = env.step(action)
            ep_states.append(state); ep_actions.append(action); ep_rewards.append(reward)
            state = next_state

        # 2. 计算折扣回报 G_t
        returns = []; G = 0
        for r in reversed(ep_rewards):
            G = r + gamma * G; returns.insert(0, G)
        returns = np.array(returns)

        # 3. 计算优势
        if use_baseline:
            values = value.forward(np.array(ep_states)).flatten()
            advantages = returns - values
        else:
            advantages = returns.copy()

        # 归一化优势（降低方差）
        if len(advantages) > 1:
            advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

        # 4. 更新策略网络
        policy.update(ep_states, ep_actions, advantages)

        # 5. 更新价值网络
        if use_baseline:
            value.update(ep_states, returns)

        total_r = sum(ep_rewards); rewards_history.append(total_r)
        if verbose and (ep + 1) % 100 == 0:
            avg = np.mean(rewards_history[-100:])
            print(f"  Episode {ep+1:>4}: avg_reward={avg:>6.1f}")

    return rewards_history


# 运行
env = SimpleCartPole()
print("=== REINFORCE without Baseline ===")
r1 = train_reinforce(env, n_episodes=500, use_baseline=False)
print(f"  Final avg: {np.mean(r1[-50:]):.1f}")

print("\n=== REINFORCE with Baseline ===")
r2 = train_reinforce(env, n_episodes=500, use_baseline=True)
print(f"  Final avg: {np.mean(r2[-50:]):.1f}")
```

### 7.3 输出示例

```
=== REINFORCE without Baseline ===
  Episode  100: avg_reward=  22.3
  Episode  200: avg_reward=  35.7
  Episode  300: avg_reward=  58.2
  Episode  400: avg_reward=  82.4
  Episode  500: avg_reward= 105.6
  Final avg: 108.3

=== REINFORCE with Baseline ===
  Episode  100: avg_reward=  35.1
  Episode  200: avg_reward=  72.8
  Episode  300: avg_reward= 118.5
  Episode  400: avg_reward= 155.3
  Episode  500: avg_reward= 168.2
  Final avg: 172.6
```

> REINFORCE with Baseline 收敛更快、最终表现更好——baseline 的方差降低效果显著。

---

## 8. 算法对比总结

### REINFORCE vs 之前学过的算法

| 维度 | Q-Learning/DQN | REINFORCE | REINFORCE+Baseline |
|------|----------------|-----------|---------------------|
| **学习对象** | $Q(s,a)$ | $\pi_\theta(a\|s)$ | $\pi_\theta(a\|s)$ + $V(s)$ |
| **策略类型** | $\varepsilon$-贪心 → 确定性 | 随机策略 | 随机策略 |
| **更新方式** | TD 误差 | MC 回报 | MC 回报 - baseline |
| **On/Off-Policy** | Off-Policy | On-Policy | On-Policy |
| **方差** | 低 | 很高 | 较高 |
| **偏差** | 有（自举） | **无偏** | **无偏** |
| **连续动作** | 难（需 $\arg\max$） | **天然支持** | **天然支持** |
| **需要回放池** | 是 | 否 | 否 |

### Value-Based vs Policy-Based 何时选择？

| 场景 | 推荐 | 原因 |
|------|------|------|
| 离散动作 + 有模型 | DP | 精确、高效 |
| 离散动作 + 大状态 | DQN | 采样效率高 |
| 连续动作 | 策略梯度 | 无需 $\arg\max$ |
| 需要随机策略 | 策略梯度 | 天然随机 |
| 竞争性环境 | 策略梯度 | 随机策略不可预测 |

---

## 9. 总结与下节预告

### 本节核心知识点

| # | 概念 | 一句话 |
|---|------|--------|
| 1 | 策略参数化 | 直接用 $\pi_\theta(a\|s)$ 表示策略，可以输出概率 |
| 2 | 对数导数技巧 | $\nabla P = P \cdot \nabla \log P$，把乘法变加法 |
| 3 | 策略梯度定理 | $\nabla J = \mathbb{E}[\nabla \log \pi(a\|s) \cdot G_t]$，不需要模型 |
| 4 | REINFORCE | MC 采样 + 梯度上升，简单但高方差 |
| 5 | Baseline | 减去 $V(s)$ 不引入偏差，但大幅降低方差 |
| 6 | 优势函数 | $A(s,a) = G_t - V(s)$，衡量动作比平均好多少 |

### 关键公式速查

| 公式 | 名称 | 用途 |
|------|------|------|
| $\pi_\theta(a\|s) = \text{softmax}(h_\theta(s,a))$ | Softmax 策略 | 离散动作 |
| $\nabla_\theta J = \mathbb{E}[\nabla \log \pi \cdot G_t]$ | 策略梯度 | REINFORCE |
| $\nabla_\theta J = \mathbb{E}[\nabla \log \pi \cdot (G_t - b)]$ | 带Baseline | 降低方差 |
| $\theta \leftarrow \theta + \alpha \nabla_\theta J$ | 梯度上升 | 参数更新 |

### 下节预告：Day 11 — Actor-Critic 方法

明天我们将把 REINFORCE 的 MC 回报替换为**TD 估计**：

- **Actor**：策略网络 $\pi_\theta$，负责选动作
- **Critic**：价值网络 $V_w$，负责评估动作好坏
- **A2C/A3C**：同步/异步多线程加速
- **GAE**：广义优势估计，更精确的优势计算

核心思想：REINFORCE 用 MC 回报 $G_t$（高方差），Actor-Critic 用 $R + \gamma V(S_{t+1})$（低方差但有偏差）——这是 Day 5 TD vs MC 思想在策略梯度中的体现！

---

## 课后练习

1. **概念题**：策略梯度方法为什么不需要环境模型 $P(s'|s,a)$？这个性质是从推导的哪一步得出的？

2. **推导题**：从 $\nabla_\theta \sum_a \pi_\theta(a|s) = 0$ 出发，证明 $\mathbb{E}_{a \sim \pi_\theta}[\nabla_\theta \log \pi_\theta(a|s)] = 0$，并说明为什么这意味着 baseline 不引入偏差。

3. **计算题**：在 2 状态 2 动作的示例中，如果 Agent 选"左"得到奖励 $R = -2$，计算 $\theta_1$ 和 $\theta_2$ 的更新量（$\alpha = 0.1$，$\theta_1 = \theta_2 = 0$）。

4. **编程题**：修改代码，实现一个用高斯策略的 REINFORCE（连续动作版本），在 Pendulum 环境上测试。

---

> **参考资料**：Sutton & Barto, Chapter 13: Policy Gradient Methods; Silver Lecture 7
