<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Day 8：函数近似与DQN - 强化学习笔记</title>
<style>
:root{--bg:#0d1117;--card:#161b22;--border:#30363d;--text:#e6edf3;--muted:#8b949e;--accent:#58a6ff;--orange:#f78166;--green:#3fb950;--purple:#bc8cff;--yellow:#d29922;--code-bg:#1c2128}
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI','Noto Sans SC',sans-serif;background:var(--bg);color:var(--text);line-height:1.8;max-width:880px;margin:0 auto;padding:40px 24px}
h1{font-size:36px;margin-bottom:8px;background:linear-gradient(135deg,var(--accent),var(--purple));-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.subtitle{color:var(--muted);font-size:16px;margin-bottom:32px}
h2{font-size:24px;margin:36px 0 16px;padding-bottom:8px;border-bottom:2px solid var(--border)}
h3{font-size:18px;margin:24px 0 10px;color:var(--accent)}
p{margin:10px 0} ul,ol{margin:8px 0 8px 24px} li{margin:4px 0}
a{color:var(--accent)}
.formula{background:var(--code-bg);border:1px solid var(--border);border-left:3px solid var(--accent);padding:14px 18px;margin:14px 0;border-radius:0 8px 8px 0;overflow-x:auto}
.formula .lbl{font-size:12px;color:var(--orange);margin-bottom:4px;font-weight:600}
.key-box{background:linear-gradient(135deg,#1a2332,#1c2a3a);border:1px solid var(--accent);border-radius:8px;padding:14px 18px;margin:14px 0}
.key-box .kbt{color:var(--accent);font-weight:600;font-size:14px;margin-bottom:8px}
table{width:100%;border-collapse:collapse;margin:14px 0;font-size:14px}
th{background:var(--code-bg);color:var(--accent);padding:10px 14px;text-align:left;border-bottom:2px solid var(--border);font-size:13px}
td{padding:10px 14px;border-bottom:1px solid var(--border)}
tr:hover td{background:#1a2233}
pre{background:var(--code-bg);border:1px solid var(--border);border-radius:8px;padding:14px;overflow-x:auto;margin:14px 0;font-family:'Fira Code','Consolas',monospace;font-size:13px;line-height:1.6}
code{font-family:'Fira Code','Consolas',monospace;background:var(--code-bg);padding:2px 6px;border-radius:4px;font-size:.9em}
blockquote{border-left:3px solid var(--purple);padding:8px 16px;margin:12px 0;color:var(--muted);font-style:italic;background:rgba(188,140,255,.05)}
.flow{display:flex;align-items:center;justify-content:center;flex-wrap:wrap;gap:8px;margin:14px 0;padding:14px;background:var(--code-bg);border-radius:8px}
.fn{background:var(--card);border:2px solid var(--accent);border-radius:8px;padding:8px 16px;font-size:14px;font-weight:600;white-space:nowrap}
.fa{color:var(--accent);font-size:20px}
.summary{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:10px;margin:14px 0}
.sc{background:var(--code-bg);border:1px solid var(--border);border-radius:8px;padding:14px}
.sc .n{display:inline-block;width:22px;height:22px;background:var(--accent);color:var(--bg);border-radius:50%;text-align:center;line-height:22px;font-size:11px;font-weight:700;margin-right:8px}
.sc .name{font-weight:600;font-size:14px}
.sc .desc{color:var(--muted);font-size:13px;margin-top:4px}
footer{text-align:center;margin-top:48px;padding-top:24px;border-top:1px solid var(--border);color:var(--muted);font-size:13px}
@media(max-width:640px){body{padding:20px 14px}h1{font-size:28px}}
</style>
<script>window.MathJax={tex:{inlineMath:[['$','$']],displayMath:[['$$','$$']]},svg:{fontCache:'global'}}</script>
<script async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
</head>
<body>
<h1>Day 8：函数近似与 DQN</h1>
<p class="subtitle">Function Approximation &amp; Deep Q-Network — 从表格到神经网络</p>

<h2>1. 核心突破：从表格到网络</h2>
<div class="formula"><div class="lbl">函数近似</div>$\underbrace{Q(s,a) \text{ 存在表格中}}_{\text{Day 1-7}} \longrightarrow \underbrace{\hat{Q}(s,a;\theta) \approx Q^*(s,a)}_{\text{Day 8+}}$</div>
<p>表格型方法无法处理大规模/连续状态空间，用参数化函数代替。</p>

<h2>2. DQN 损失函数</h2>
<div class="formula"><div class="lbl">均方 TD 误差</div>$L(\theta) = \mathbb{E}_{(s,a,r,s')} \Big[ \big( \underbrace{r + \gamma \max_{a'} \hat{Q}(s',a';\theta^-)}_{\text{TD 目标 } y} - \hat{Q}(s,a;\theta) \big)^2 \Big]$</div>
<div class="formula"><div class="lbl">梯度更新</div>$\theta \leftarrow \theta + \alpha \big[ y - \hat{Q}(s,a;\theta) \big] \nabla_\theta \hat{Q}(s,a;\theta)$</div>
<table>
<tr><th>符号</th><th>含义</th></tr>
<tr><td>$\hat{Q}(s,a;\theta)$</td><td>主网络预测的 Q 值</td></tr>
<tr><td>$y = r + \gamma \max_{a'} Q(s',a';\theta^-)$</td><td>目标网络计算的 TD 目标</td></tr>
<tr><td>$\theta^-$</td><td>目标网络参数（不参与梯度）</td></tr>
</table>

<h2>3. 两大挑战与解决方案</h2>
<table>
<tr><th>挑战</th><th>问题</th><th>解决方案</th></tr>
<tr><td>样本相关性</td><td>连续数据高度相关</td><td><strong>经验回放</strong>：随机采样打破相关性</td></tr>
<tr><td>训练不稳定</td><td>目标依赖 $\theta$，追着自己跑</td><td><strong>目标网络</strong>：$\theta^-$ 冻结目标</td></tr>
</table>

<h2>4. 经验回放</h2>
<div class="key-box"><div class="kbt">核心思想</div>
<p>把交互数据存入回放池 $\mathcal{D}$，训练时随机采样 mini-batch，打破时间相关性。</p>
<p>三个好处：① 打破相关性 ② 提高数据效率 ③ 可用旧策略数据（Off-Policy）</p></div>

<h2>5. 目标网络</h2>
<div class="formula"><div class="lbl">硬更新 (每 C 步)</div>$\theta^- \leftarrow \theta \quad \text{每 } C \text{ 步}$</div>
<div class="formula"><div class="lbl">软更新 (每步)</div>$\theta^- \leftarrow \tau \theta + (1 - \tau) \theta^- \quad (\tau \ll 1)$</div>
<div class="flow">
<span class="fn">主网络 θ<br/>快速更新</span><span class="fa">→</span><span class="fn">目标网络 θ⁻<br/>缓慢追踪</span><span class="fa">→</span><span class="fn">目标 y 稳定</span><span class="fa">→</span><span class="fn">训练收敛</span>
</div>

<h2>6. DQN 算法流程</h2>
<pre>1. 初始化主网络 θ, 目标网络 θ⁻ ← θ, 回放池 D
2. for each episode:
3.   while not done:
4.     ε-贪心选动作 a
5.     执行 a, 观察 r, s'
6.     D.store((s, a, r, s'))
7.     从 D 随机采样 mini-batch
8.     计算 y = r + γ max Q(s',a'; θ⁻)
9.     梯度下降更新 θ
10.    每 C 步: θ⁻ ← θ</pre>

<h2>7. CartPole 实战结果</h2>
<pre>状态: [位置, 速度, 角度, 角速度] (连续 4 维)
动作: 左/右 (离散 2 个)
目标: 杆不倒, 每步 +1

Episode  50 | 平均奖励:   22.3 | ε: 0.395
Episode 100 | 平均奖励:   85.7 | ε: 0.136
Episode 200 | 平均奖励:  243.1 | ε: 0.024
Episode 300 | 平均奖励:  385.2 | ε: 0.011

最终: 385.2 (解决标准 ≥ 195) ✓</pre>

<h2>8. DQN vs Q-Learning</h2>
<table>
<tr><th>维度</th><th>Q-Learning</th><th>DQN</th></tr>
<tr><td>存储</td><td>表格</td><td>神经网络</td></tr>
<tr><td>状态空间</td><td>小规模离散</td><td>任意（含连续）</td></tr>
<tr><td>泛化</td><td>无</td><td>有（相似状态共享）</td></tr>
<tr><td>数据使用</td><td>一次</td><td>经验回放反复使用</td></tr>
<tr><td>稳定性</td><td>稳定</td><td>需目标网络</td></tr>
</table>

<h2>9. 核心总结</h2>
<div class="summary">
<div class="sc"><span class="n">1</span><span class="name">函数近似</span><div class="desc">$\hat{Q}(s,a;\theta) \approx Q^*$，参数代替表格</div></div>
<div class="sc"><span class="n">2</span><span class="name">经验回放</span><div class="desc">随机采样打破相关性</div></div>
<div class="sc"><span class="n">3</span><span class="name">目标网络</span><div class="desc">$\theta^-$ 冻结目标，稳定训练</div></div>
<div class="sc"><span class="n">4</span><span class="name">DQN 损失</span><div class="desc">$L = (y - \hat{Q})^2$</div></div>
<div class="sc"><span class="n">5</span><span class="name">软更新</span><div class="desc">$\theta^- \leftarrow \tau\theta + (1-\tau)\theta^-$</div></div>
</div>

<div class="formula" style="border-left-color:var(--purple);margin-top:24px">
<div class="lbl"> 下节预告：Day 9 — DQN 改进</div>
<p style="margin:0">Double DQN（消除过估计）、Dueling DQN（分离 V 和 A）、PER（优先采样）</p>
</div>

<footer>
<p>强化学习系统学习笔记 · Day 8 · <a href="https://github.com/Xinyu-lumos/RL-study">GitHub 仓库</a></p>
<p>参考：Mnih et al., Nature 2015</p>
</footer>
</body>
</html>
