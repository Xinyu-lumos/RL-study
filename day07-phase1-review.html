<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Day 7：第一阶段复习与实践 - 强化学习笔记</title>
<style>
:root{--bg:#0d1117;--card:#161b22;--border:#30363d;--text:#e6edf3;--muted:#8b949e;--accent:#58a6ff;--orange:#f78166;--green:#3fb950;--purple:#bc8cff;--yellow:#d29922;--code-bg:#1c2128}
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI','Noto Sans SC',sans-serif;background:var(--bg);color:var(--text);line-height:1.8;max-width:880px;margin:0 auto;padding:40px 24px}
h1{font-size:36px;margin-bottom:8px;background:linear-gradient(135deg,var(--accent),var(--purple));-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.subtitle{color:var(--muted);font-size:16px;margin-bottom:32px}
h2{font-size:24px;margin:36px 0 16px;padding-bottom:8px;border-bottom:2px solid var(--border)}
h3{font-size:18px;margin:24px 0 10px;color:var(--accent)}
h4{font-size:15px;margin:18px 0 8px;color:var(--purple)}
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
.tag{display:inline-block;padding:2px 8px;border-radius:12px;font-size:12px;font-weight:600}
.tg-on{background:#1a3a2a;color:var(--green);border:1px solid var(--green)}
.tg-off{background:#3a2a1a;color:var(--orange);border:1px solid var(--orange)}
.tg-dp{background:#2a1a3a;color:var(--purple);border:1px solid var(--purple)}
.tg-td{background:#3a3a1a;color:var(--yellow);border:1px solid var(--yellow)}
footer{text-align:center;margin-top:48px;padding-top:24px;border-top:1px solid var(--border);color:var(--muted);font-size:13px}
@media(max-width:640px){body{padding:20px 14px}h1{font-size:28px}}
</style>
<script>window.MathJax={tex:{inlineMath:[['$','$']],displayMath:[['$$','$$']]},svg:{fontCache:'global'}}</script>
<script async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
</head>
<body>
<h1>Day 7：第一阶段复习与实践</h1>
<p class="subtitle">Phase 1 Review — 串联 MDP → DP → MC → TD → SARSA/Q-Learning</p>

<h2>1. 知识全景：六天路线</h2>
<div class="flow">
<span class="fn">MDP+贝尔曼</span><span class="fa">→</span><span class="fn">DP</span><span class="fa">→</span><span class="fn">MC</span><span class="fa">→</span><span class="fn">TD</span><span class="fa">→</span><span class="fn">SARSA/Q</span>
</div>
<table>
<tr><th>天数</th><th>核心问题</th><th>关键突破</th></tr>
<tr><td>Day 1-2</td><td>怎么建模？价值怎么递推？</td><td>MDP + 贝尔曼方程</td></tr>
<tr><td>Day 3</td><td>有模型怎么求解？</td><td>策略迭代 / 价值迭代</td></tr>
<tr><td>Day 4</td><td>没模型怎么办？</td><td>MC：采样取平均</td></tr>
<tr><td>Day 5</td><td>怎么不等回合结束？</td><td>TD：每步更新</td></tr>
<tr><td>Day 6</td><td>怎么找最优策略？</td><td>SARSA / Q-Learning</td></tr>
</table>

<h2>2. 核心公式速查</h2>
<h3>基础定义</h3>
<div class="formula"><div class="lbl">回报递推</div>$G_t = R_{t+1} + \gamma G_{t+1}$</div>
<div class="formula"><div class="lbl">V-Q 关系</div>$V^\pi(s) = \sum_a \pi(a|s) Q^\pi(s,a)$</div>

<h3>贝尔曼方程</h3>
<table>
<tr><th>名称</th><th>公式</th><th>用途</th></tr>
<tr><td>期望方程</td><td>$V^\pi(s) = \sum_a \pi \sum_{s'} P[R+\gamma V^\pi(s')]$</td><td>评估策略</td></tr>
<tr><td>最优方程</td><td>$V^*(s) = \max_a \sum_{s'} P[R+\gamma V^*(s')]$</td><td>找最优</td></tr>
</table>

<h3>更新规则</h3>
<table>
<tr><th>算法</th><th>更新公式</th><th>标签</th></tr>
<tr><td>MC</td><td>$V \leftarrow V + \alpha(G - V)$</td><td><span class="tag tg-on">On</span> 无自举</td></tr>
<tr><td>TD(0)</td><td>$V \leftarrow V + \alpha[R+\gamma V' - V]$</td><td><span class="tag tg-td">TD</span> 自举</td></tr>
<tr><td>SARSA</td><td>$Q \leftarrow Q + \alpha[R+\gamma Q(S',A') - Q]$</td><td><span class="tag tg-on">On</span></td></tr>
<tr><td>Q-Learning</td><td>$Q \leftarrow Q + \alpha[R+\gamma\max Q - Q]$</td><td><span class="tag tg-off">Off</span></td></tr>
</table>

<h3>统一更新视角</h3>
<div class="formula"><div class="lbl">一切 TD 方法的统一形式</div>$Q(S,A) \leftarrow Q(S,A) + \alpha \big[ \underbrace{R + \gamma U}_{\text{TD 目标}} - Q(S,A) \big]$</div>
<p>$U$ 决定算法：$U=Q(S',A')$ → SARSA；$U=\sum\pi Q$ → Expected SARSA；$U=\max Q$ → Q-Learning</p>

<h2>3. 算法关系图谱</h2>
<div class="flow">
<span class="fn">有模型?</span><span class="fa">→</span><span class="fn">DP</span><span class="fa">→</span><span class="fn">无模型?</span><span class="fa">→</span><span class="fn">MC/TD</span><span class="fa">→</span><span class="fn">On/Off?</span>
</div>
<table>
<tr><th>维度</th><th>DP</th><th>MC</th><th>TD/SARSA</th><th>Q-Learning</th></tr>
<tr><td>需要模型</td><td>Yes</td><td>No</td><td>No</td><td>No</td></tr>
<tr><td>自举</td><td>Yes</td><td>No</td><td>Yes</td><td>Yes</td></tr>
<tr><td>等回合结束</td><td>No</td><td>Yes</td><td>No</td><td>No</td></tr>
<tr><td>On/Off</td><td>—</td><td>On</td><td>On</td><td>Off</td></tr>
</table>

<h2>4. 关键概念辨析</h2>
<h3>On-Policy vs Off-Policy</h3>
<table>
<tr><th></th><th>On-Policy <span class="tag tg-on">同</span></th><th>Off-Policy <span class="tag tg-off">异</span></th></tr>
<tr><td>采样策略</td><td>= 目标策略</td><td>≠ 目标策略</td></tr>
<tr><td>代表</td><td>SARSA, MC</td><td>Q-Learning</td></tr>
<tr><td>最终策略</td><td>ε-最优</td><td>纯贪心最优</td></tr>
</table>

<h3>偏差-方差谱</h3>
<div class="key-box"><div class="kbt">MC → n-step TD → TD(0)</div>
<p>MC（无偏，高方差）←→ TD(0)（有偏，低方差）<br>
n-step TD 在两者间插值：$n=1$ 是 TD(0)，$n=\infty$ 是 MC</p></div>

<h2>5. 综合实战：四种算法横评</h2>
<pre>Cliff Walking 环境, α=0.1, γ=1.0, ε=0.1, 500回合×50次平均

最后 50 回合平均奖励:
  Expected SARSA: -78.5  (最好: 低方差 + 考虑风险)
  SARSA:          -90.2  (安全策略, 保守绕行)
  MC Control:     -105.3 (高方差, 收敛慢)
  Q-Learning:     -135.7 (学到最优路径但 ε-贪心执行掉崖)</pre>

<h2>6. GPI：一切算法的母框架</h2>
<div class="flow">
<span class="fn">策略 π</span><span class="fa">→评估 E→</span><span class="fn">价值 V/Q</span><span class="fa">→改进 I→</span><span class="fn">策略 π</span>
</div>
<table>
<tr><th>算法</th><th>E（评估）</th><th>I（改进）</th></tr>
<tr><td>策略迭代</td><td>贝尔曼期望迭代到收敛</td><td>完全贪心</td></tr>
<tr><td>价值迭代</td><td>贝尔曼最优一步</td><td>max 隐式贪心</td></tr>
<tr><td>MC 控制</td><td>采样取平均</td><td>ε-贪心</td></tr>
<tr><td>SARSA</td><td>TD 更新 Q</td><td>ε-贪心</td></tr>
<tr><td>Q-Learning</td><td>TD 更新 Q</td><td>max (Off-Policy)</td></tr>
</table>

<h2>7. 第二阶段展望：深度强化学习</h2>
<div class="key-box"><div class="kbt">核心突破：用神经网络代替表格</div>
<p>$V(s)$ 存表格 → $V(s;\theta) = \text{NN}(s)$ 函数近似</p>
<p>DQN 桥接：$Q(s,a) \approx Q(s,a;\theta)$，用梯度下降更新</p></div>
<table>
<tr><th>天数</th><th>主题</th><th>突破</th></tr>
<tr><td>Day 8</td><td>DQN</td><td>神经网络 + 经验回放 + 目标网络</td></tr>
<tr><td>Day 9</td><td>DQN 改进</td><td>Double/Dueling DQN, PER</td></tr>
<tr><td>Day 10</td><td>策略梯度</td><td>直接优化 $\pi_\theta(a|s)$</td></tr>
<tr><td>Day 11</td><td>Actor-Critic</td><td>同时优化策略和价值</td></tr>
<tr><td>Day 12</td><td>PPO</td><td>最流行的稳定策略梯度</td></tr>
<tr><td>Day 13</td><td>SAC/TD3</td><td>连续动作空间 SOTA</td></tr>
</table>

<h2>8. 必须记住的 5 个核心</h2>
<div class="summary">
<div class="sc"><span class="n">1</span><span class="name">贝尔曼递推</span><div class="desc">$G_t = R_{t+1} + \gamma G_{t+1}$ 无限→一步</div></div>
<div class="sc"><span class="n">2</span><span class="name">GPI</span><div class="desc">评估+改进交替，统一框架</div></div>
<div class="sc"><span class="n">3</span><span class="name">MC vs TD</span><div class="desc">无偏高方差 vs 有偏低方差</div></div>
<div class="sc"><span class="n">4</span><span class="name">SARSA vs Q-Learning</span><div class="desc">$Q(S',A')$ vs $\max Q$</div></div>
<div class="sc"><span class="n">5</span><span class="name">统一更新</span><div class="desc">$Q \leftarrow Q + \alpha[\text{TD目标} - Q]$</div></div>
</div>

<footer>
<p>强化学习系统学习笔记 · Day 7 · <a href="https://github.com/Xinyu-lumos/RL-study">GitHub 仓库</a></p>
<p>参考：Sutton &amp; Barto, Chapters 3-6</p>
</footer>
</body>
</html>
