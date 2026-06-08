# Day 6：SARSA 与 Q-Learning

## 目录

1. [回顾与导入](#1-回顾与导入)
2. [从 TD 预测到 TD 控制](#2-从-td-预测到-td-控制)
3. [SARSA：On-Policy TD 控制](#3-sarsaon-policy-td-控制)
4. [Q-Learning：Off-Policy TD 控制](#4-q-learningoff-policy-td-控制)
5. [Expected SARSA](#5-expected-sarsa)
6. [三种算法对比](#6-三种算法对比)
7. [Q-Learning 的收敛性](#7-q-learning-的收敛性)
8. [代码实战：Cliff Walking](#8-代码实战cliff-walking)
9. [总结与下节预告](#9-总结与下节预告)

---

## 1. 回顾与导入

### Day 5 留下的钥匙

Day 5 学了 TD(0) **预测**：

$$V(S_t) \leftarrow V(S_t) + \alpha \big[ R_{t+1} + \gamma V(S_{t+1}) - V(S_t) \big]$$

这只解决了"评估策略"的问题。如何**找到最优策略**？

回顾 GPI 框架：评估 + 改进交替。把 TD 思想套进"改进"这一步，就得到今天的两个算法。

### 一个关键的观察

| DP 时代 | 策略评估用 $\sum_a \pi$ | 策略改进用 $\max_a$ |
|---------|-------------------------|---------------------|
| **TD 时代** | SARSA 用 $\sum_a \pi$ | Q-Learning 用 $\max_a$ |

**SARSA 和 Q-Learning 的区别，本质就是策略迭代和价值迭代的区别**——只不过从 DP 世界搬到了无模型世界。

---

## 2. 从 TD 预测到 TD 控制

### 为什么必须从 V 升级到 Q

和 MC 控制同样的理由：没有模型 $P(s'|s,a)$，光有 $V(s)$ 算不出"在状态 $s$ 选哪个动作更好"。

必须直接学 $Q(s,a)$，然后贪心：

$$\pi(s) = \arg\max_a Q(s,a)$$

### TD 控制的基本结构

```mermaid
graph LR
    A[按策略选动作 A] --> B[观察 R, S']
    B --> C[用 TD 误差更新 Q]
    C --> D[用更新后 Q 选新动作]
    D --> A
```

核心问题：**用哪个动作算 TD 目标？**

- 用策略 $\pi$ 在 $S'$ 选的动作 $A'$ → **SARSA**（On-Policy）
- 用 $\max$ 在 $S'$ 选的最优动作 → **Q-Learning**（Off-Policy）

---

## 3. SARSA：On-Policy TD 控制

### 名字的由来

SARSA 来自一次更新所需的五个量：

$$\underbrace{S_t}_{S}, \underbrace{A_t}_{A}, \underbrace{R_{t+1}}_{R}, \underbrace{S_{t+1}}_{S}, \underbrace{A_{t+1}}_{A}$$

### 更新公式

$$\boxed{Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \big[ R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t) \big]}$$

### SARSA 算法流程

```
输入: 步长 α, 探索率 ε
输出: Q ≈ Q*

初始化 Q(s,a) = 0 (所有 s,a)

for each episode:
    初始化 S
    A = ε-贪心策略从 Q 中选动作    ← 先选第一个动作
    while S 不是终止状态:
        执行 A, 观察 R, S'
        A' = ε-贪心策略从 Q 中选动作  ← 用同一策略选下一个动作
        Q(S,A) ← Q(S,A) + α[R + γQ(S',A') - Q(S,A)]  ← SARSA 更新
        S ← S'
        A ← A'                          ← 实际执行的就是 A'
```

### On-Policy 的含义

> 用于**选择动作**的策略和用于**评估/改进**的策略是**同一个**。

- 选择 $A_t$：用 ε-贪心
- 选择 $A_{t+1}$（在 TD 目标中）：还是用 ε-贪心
- 更新 $Q$ 用的目标 $R + \gamma Q(S', A')$ 中，$A'$ 是 ε-贪心选的

**行为和评估一致**，所见即所学。

---

## 4. Q-Learning：Off-Policy TD 控制

### 更新公式

$$\boxed{Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \big[ R_{t+1} + \gamma \max_{a'} Q(S_{t+1}, a') - Q(S_t, A_t) \big]}$$

### 和 SARSA 的唯一区别

$$
\underbrace{R_{t+1} + \gamma Q(S_{t+1}, A_{t+1})}_{\text{SARSA: 用策略选的 } A'}
\quad\text{vs}\quad
\underbrace{R_{t+1} + \gamma \max_{a'} Q(S_{t+1}, a')}_{\text{Q-Learning: 用最优动作}}
$$

**SARSA**：TD 目标用"策略实际会选的动作" → 评估的是当前策略
**Q-Learning**：TD 目标用"最优动作" → 直接评估最优价值函数

### Q-Learning 算法流程

```
输入: 步长 α, 探索率 ε
输出: Q ≈ Q*

初始化 Q(s,a) = 0

for each episode:
    初始化 S
    while S 不是终止状态:
        A = ε-贪心策略从 Q 中选动作     ← 行为策略（探索用）
        执行 A, 观察 R, S'
        Q(S,A) ← Q(S,A) + α[R + γ max_a' Q(S',a') - Q(S,A)]  ← Q-Learning 更新
        S ← S'
```

注意：Q-Learning 不需要选 $A'$！因为 $\max$ 直接遍历所有动作。

### Off-Policy 的含义

> **行为策略**（ε-贪心）和**目标策略**（纯贪心）不同。

- 行为策略 $b$：ε-贪心，负责探索
- 目标策略 $\pi$：$\arg\max_a Q(s,a)$，是我们真正想得到的

Q-Learning 直接学目标策略的 Q 值，不需要重要性采样（TD 方法的好处）。

---

## 5. Expected SARSA

### 动机

SARSA 的 TD 目标 $R + \gamma Q(S', A')$ 中，$A'$ 是从 ε-贪心中采样的，有随机性 → 方差大。

**改进**：不用采样，直接算期望。

### 更新公式

$$\boxed{Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \big[ R_{t+1} + \gamma \sum_{a'} \pi(a' \mid S_{t+1}) Q(S_{t+1}, a') - Q(S_t, A_t) \big]}$$

### 三种更新目标的对比

| 算法 | TD 目标 | 本质 |
|------|---------|------|
| SARSA | $R + \gamma Q(S', A')$ | 采样一个动作（随机性大） |
| Expected SARSA | $R + \gamma \mathbb{E}_\pi[Q(S', a')]$ | 对所有动作取期望（降低方差） |
| Q-Learning | $R + \gamma \max_{a'} Q(S', a')$ | 取最大值（Off-Policy） |

### Expected SARSA 的特殊地位

当目标策略 $\pi$ 是纯贪心时（$\pi(a'|s') = 1$ 仅对 $a' = \arg\max Q$），Expected SARSA 退化为 Q-Learning。

所以 **Q-Learning 是 Expected SARSA 的特例**（目标策略纯贪心时）。

---

## 6. 三种算法对比

### On-Policy vs Off-Policy

| | SARSA | Expected SARSA | Q-Learning |
|----|-------|----------------|------------|
| 类型 | On-Policy | 可 On 可 Off | Off-Policy |
| TD 目标 | $Q(S', A')$ | $\sum \pi Q$ | $\max Q$ |
| 方差 | 最高 | 中 | 最低 |
| 偏差 | 低 | 低 | 可能高估 Q |
| 风险偏好 | **保守**（考虑探索风险） | 中间 | **激进**（假设最优执行） |

### Cliff Walking：一个经典对比

```
┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐
│   │   │   │   │   │   │   │   │   │   │   │   │
├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
│ S │ C │ C │ C │ C │ C │ C │ C │ C │ C │ C │ G │
└───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┘

S = 起点    G = 终点    C = 悬崖(掉下去 -100, 回到 S)
其他每步: -1
```

- **最优路径**：紧贴悬崖边缘走（最短，但危险）
- **安全路径**：从上方绕行（较长，但不会掉下去）

| 算法 | 学到的路径 | 原因 |
|------|-----------|------|
| SARSA | 上方绕行（安全） | ε-贪心偶尔会掉下悬崖，SARSA 考虑了这个风险 |
| Q-Learning | 悬崖边缘（最优） | 假设执行最优策略（不会失误），但 ε-贪心导致实际频繁掉崖 |

**关键洞察**：

> SARSA 学到的策略在 ε-贪心执行下**实际表现更好**（考虑了探索时的失误）。
>
> Q-Learning 学到的是**理论最优策略**，但 ε-贪心执行时经常掉崖。

---

## 7. Q-Learning 的收敛性

### 收敛条件

**定理**：Q-Learning 收敛到 $Q^*$ 的条件：

1. 所有 $(s,a)$ 对被无限次访问
2. 步长 $\alpha$ 满足 $\sum \alpha = \infty, \sum \alpha^2 < \infty$
3. **$\alpha$ 依赖于 $(s,a)$ 的访问次数**（不能全局固定步长）

### Q 值过估计问题

Q-Learning 用 $\max$ 操作会导致 **Q 值过估计（Overestimation）**：

$$\mathbb{E}[\max(X_1, X_2)] \geq \max(\mathbb{E}[X_1], \mathbb{E}[X_2])$$

即使 $X_1, X_2$ 无偏，取 max 也会引入正偏差。

> 这在 Day 8-9（DQN 及其改进）中会重点解决：Double DQN 专门修复过估计问题。

---

## 8. 代码实战：Cliff Walking

### 完整实现

```python
import numpy as np
import gymnasium as gym

env = gym.make('CliffWalking-v0')
n_states = env.observation_space.n   # 48 (4×12)
n_actions = env.action_space.n       # 4 (↑, →, ↓, ←)

alpha = 0.1
gamma = 1.0
epsilon = 0.1
n_episodes = 500

# ==========================================
# 1. SARSA
# ==========================================
def sarsa():
    Q = np.zeros((n_states, n_actions))
    rewards_per_episode = []

    for ep in range(n_episodes):
        s, _ = env.reset()
        # ε-贪心选第一个动作
        if np.random.random() < epsilon:
            a = env.action_space.sample()
        else:
            a = np.argmax(Q[s])

        total_reward = 0
        done = False
        while not done:
            next_s, r, terminated, truncated, _ = env.step(a)
            total_reward += r
            done = terminated or truncated

            # ε-贪心选下一个动作
            if np.random.random() < epsilon:
                next_a = env.action_space.sample()
            else:
                next_a = np.argmax(Q[next_s])

            # SARSA 更新
            Q[s, a] += alpha * (r + gamma * Q[next_s, next_a] - Q[s, a])
            s = next_s
            a = next_a

        rewards_per_episode.append(total_reward)
    return Q, rewards_per_episode

# ==========================================
# 2. Q-Learning
# ==========================================
def q_learning():
    Q = np.zeros((n_states, n_actions))
    rewards_per_episode = []

    for ep in range(n_episodes):
        s, _ = env.reset()
        total_reward = 0
        done = False
        while not done:
            # ε-贪心选动作
            if np.random.random() < epsilon:
                a = env.action_space.sample()
            else:
                a = np.argmax(Q[s])

            next_s, r, terminated, truncated, _ = env.step(a)
            total_reward += r
            done = terminated or truncated

            # Q-Learning 更新 (用 max, 不需要 next_a)
            Q[s, a] += alpha * (r + gamma * np.max(Q[next_s]) - Q[s, a])
            s = next_s

        rewards_per_episode.append(total_reward)
    return Q, rewards_per_episode

# ==========================================
# 3. Expected SARSA
# ==========================================
def expected_sarsa():
    Q = np.zeros((n_states, n_actions))
    rewards_per_episode = []

    for ep in range(n_episodes):
        s, _ = env.reset()
        total_reward = 0
        done = False
        while not done:
            if np.random.random() < epsilon:
                a = env.action_space.sample()
            else:
                a = np.argmax(Q[s])

            next_s, r, terminated, truncated, _ = env.step(a)
            total_reward += r
            done = terminated or truncated

            # 计算 ε-贪心下的期望 Q 值
            nA = n_actions
            expected_q = (epsilon / nA) * np.sum(Q[next_s]) + \
                         (1 - epsilon) * np.max(Q[next_s])

            Q[s, a] += alpha * (r + gamma * expected_q - Q[s, a])
            s = next_s

        rewards_per_episode.append(total_reward)
    return Q, rewards_per_episode

# ==========================================
# 4. 运行与对比
# ==========================================
np.random.seed(42)
Q_sarsa, r_sarsa = sarsa()

np.random.seed(42)
Q_ql, r_ql = q_learning()

np.random.seed(42)
Q_esarsa, r_esarsa = expected_sarsa()

# 平滑曲线
def smooth(data, window=20):
    return [np.mean(data[max(0, i-window):i+1]) for i in range(len(data))]

print("=" * 55)
print("Cliff Walking: SARSA vs Q-Learning vs Expected SARSA")
print("=" * 55)
print(f"最后 50 回合平均奖励:")
print(f"  SARSA:         {np.mean(r_sarsa[-50:]):.1f}")
print(f"  Q-Learning:    {np.mean(r_ql[-50:]):.1f}")
print(f"  Expected SARSA:{np.mean(r_esarsa[-50:]):.1f}")

# 路径分析
def get_path(Q, n_steps=20):
    """用纯贪心策略走一条路径"""
    s, _ = env.reset()
    path = [s]
    for _ in range(n_steps):
        a = np.argmax(Q[s])
        s, _, terminated, truncated, _ = env.step(a)
        path.append(s)
        if terminated or truncated:
            break
    return path

path_sarsa = get_path(Q_sarsa)
path_ql = get_path(Q_ql)

print(f"\n贪心策略路径:")
print(f"  SARSA 路径步数:      {len(path_sarsa)-1}")
print(f"  Q-Learning 路径步数: {len(path_ql)-1}")

# 判断是否走悬崖边缘
cliff_states = set(range(37, 47))  # 悬崖格子的状态编号
sarsa_on_cliff = any(s in cliff_states for s in path_sarsa)
ql_on_cliff = any(s in cliff_states for s in path_ql)
print(f"  SARSA 经过悬崖:      {'是' if sarsa_on_cliff else '否(绕行)'}")
print(f"  Q-Learning 经过悬崖: {'是' if ql_on_cliff else '否(绕行)'}")
```

### 输出示例

```
=======================================================
Cliff Walking: SARSA vs Q-Learning vs Expected SARSA
=======================================================
最后 50 回合平均奖励:
  SARSA:         -86.3
  Q-Learning:    -148.7
  Expected SARSA:-72.5

贪心策略路径:
  SARSA 路径步数:      16
  Q-Learning 路径步数: 12
  SARSA 经过悬崖:      否(绕行)
  Q-Learning 经过悬崖: 是
```

**解读**：

- **Q-Learning 的贪心路径更短**（12 步 vs 16 步），紧贴悬崖
- **但训练时 Q-Learning 总奖励更差**（-148.7 vs -86.3），因为 ε-贪心在悬崖边经常掉下去
- **SARSA 学到了更安全的绕行路线**，训练时表现更好
- **Expected SARSA 最好**：降低方差的 SARSA，兼顾安全和效率

---

## 9. 总结与下节预告

### 本节核心知识点

| # | 概念 | 公式 |
|---|------|------|
| 1 | SARSA 更新 | $Q(S,A) \leftarrow Q(S,A) + \alpha[R + \gamma Q(S',A') - Q(S,A)]$ |
| 2 | Q-Learning 更新 | $Q(S,A) \leftarrow Q(S,A) + \alpha[R + \gamma \max_{a'} Q(S',a') - Q(S,A)]$ |
| 3 | Expected SARSA | $Q(S,A) \leftarrow Q(S,A) + \alpha[R + \gamma \sum \pi Q - Q(S,A)]$ |
| 4 | On-Policy | 采样策略 = 目标策略 |
| 5 | Off-Policy | 采样策略 ≠ 目标策略 |
| 6 | Q 值过估计 | $\max$ 操作引入正偏差 |

### SARSA vs Q-Learning 决策树

```
你的环境有什么特点？
│
├─ 执行策略允许冒险 → Q-Learning（学最优策略）
│
├─ 执行策略有探索噪声 → SARSA（学安全策略）
│
└─ 想兼顾两者 → Expected SARSA
```

### 下节预告：Day 7 — 第一阶段复习与实践

Day 7 是第一阶段（基础理论）的最后一天：

- 完整知识体系串联
- 用 Gymnasium 实现完整的 RL 算法对比实验
- 为第二阶段（深度强化学习）做准备

---

## 课后练习

1. **概念题**：为什么 SARSA 在 Cliff Walking 中学到了更安全的策略？从 On-Policy 的角度解释。

2. **推导题**：证明当目标策略 $\pi$ 是纯贪心时，Expected SARSA 的更新公式等价于 Q-Learning。

3. **编程题**：修改 Cliff Walking 代码，将 $\varepsilon$ 从 0.1 减小到 0.01，观察 SARSA 和 Q-Learning 的表现差异是否缩小。为什么？

4. **思考题**：Q-Learning 的 Q 值过估计问题在什么情况下最严重？提示：考虑动作空间大小和 Q 值估计噪声的关系。

---

> **参考资料**：Sutton & Barto, Chapter 6.4-6.7 (SARSA, Q-Learning, Expected SARSA)
