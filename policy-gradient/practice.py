"""
Day 10: Policy Gradient (REINFORCE) Practice

1. REINFORCE without baseline
2. REINFORCE with baseline (value network)
3. Variance comparison
4. Softmax policy gradient visualization

Run: python practice.py
Depends: numpy
"""

import sys; sys.stdout.reconfigure(encoding="utf-8")
import numpy as np


class SimpleCartPole:
    """CartPole-v1 simplified"""
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


class PolicyNet:
    """Softmax policy network (pure NumPy)"""
    def __init__(self, obs_dim=4, n_act=2, hid=64, lr=0.003):
        self.lr=lr; self.n_act=n_act
        self.W1=np.random.randn(obs_dim,hid)*np.sqrt(2/obs_dim); self.b1=np.zeros(hid)
        self.W2=np.random.randn(hid,n_act)*np.sqrt(2/hid);      self.b2=np.zeros(n_act)
    def forward(self, x):
        self.x=np.atleast_2d(x)
        self.h1=np.maximum(0,self.x@self.W1+self.b1)
        self.logits=self.h1@self.W2+self.b2
        e=np.exp(self.logits-np.max(self.logits,axis=1,keepdims=True))
        self.probs=e/np.sum(e,axis=1,keepdims=True)
        return self.probs
    def sample(self, state):
        p=self.forward(state)[0]; return np.random.choice(self.n_act,p=p)
    def update(self, states, actions, advs):
        self.forward(np.array(states)); bs=len(states)
        dl=self.probs.copy()
        for i in range(bs):
            dl[i,actions[i]]-=1; dl[i]*=advs[i]
        dl/=bs; clip=10.0
        dW2=self.h1.T@dl; db2=np.sum(dl,axis=0)
        dh=dl@self.W2.T*(self.h1>0); dW1=self.x.T@dh; db1=np.sum(dh,axis=0)
        for g in[dW1,db1,dW2,db2]: np.clip(g,-clip,clip,out=g)
        self.W2-=self.lr*dW2; self.b2-=self.lr*db2; self.W1-=self.lr*dW1; self.b1-=self.lr*db1


class ValueNet:
    """State value network V(s) for baseline"""
    def __init__(self, obs_dim=4, hid=64, lr=0.01):
        self.lr=lr
        self.W1=np.random.randn(obs_dim,hid)*np.sqrt(2/obs_dim); self.b1=np.zeros(hid)
        self.W2=np.random.randn(hid,1)*np.sqrt(2/hid);           self.b2=np.zeros(1)
    def forward(self, x):
        self.x=np.atleast_2d(x); self.h1=np.maximum(0,self.x@self.W1+self.b1)
        self.v=self.h1@self.W2+self.b2; return self.v
    def update(self, states, returns):
        self.forward(np.array(states)); bs=len(states)
        e=(self.v.flatten()-np.array(returns)).reshape(-1,1)
        dW2=self.h1.T@e/bs; db2=np.mean(e,axis=0)
        dh=e@self.W2.T*(self.h1>0); dW1=self.x.T@dh/bs; db1=np.mean(dh,axis=0)
        clip=10.0
        for g in[dW1,db1,dW2,db2]: np.clip(g,-clip,clip,out=g)
        self.W2-=self.lr*dW2; self.b2-=self.lr*db2; self.W1-=self.lr*dW1; self.b1-=self.lr*db1


def train(env, n_ep=500, gamma=0.99, p_lr=0.003, v_lr=0.01, baseline=True):
    policy=PolicyNet(lr=p_lr)
    value=ValueNet(lr=v_lr) if baseline else None
    rewards=[]
    for ep in range(n_ep):
        s=env.reset(); S,A,R=[],[],[]; done=False
        while not done:
            a=policy.sample(s); ns,r,done=env.step(a)
            S.append(s); A.append(a); R.append(r); s=ns
        # compute returns
        G=0; rets=[]
        for r in reversed(R): G=r+gamma*G; rets.insert(0,G)
        rets=np.array(rets)
        # compute advantages
        if baseline:
            vals=value.forward(np.array(S)).flatten(); adv=rets-vals
        else:
            adv=rets.copy()
        if len(adv)>1: adv=(adv-adv.mean())/(adv.std()+1e-8)
        # update
        policy.update(S,A,adv)
        if baseline: value.update(S,rets)
        rewards.append(sum(R))
        if(ep+1)%100==0: print(f"  Ep {ep+1:>4}: avg={np.mean(rewards[-100:]):>6.1f}")
    return rewards


def demo_softmax_gradient():
    """Demonstrate softmax policy gradient computation"""
    print("=" * 60)
    print("Demo: Softmax Policy Gradient Step-by-Step")
    print("=" * 60)
    theta = np.array([0.0, 0.0])  # [left, right] logits
    print(f"\nInitial theta = {theta}")
    # softmax
    exp_t = np.exp(theta - theta.max())
    probs = exp_t / exp_t.sum()
    print(f"pi(left) = {probs[0]:.3f}, pi(right) = {probs[1]:.3f}")

    # Agent picks "right", reward = +3
    action = 1; reward = 3; alpha = 0.1
    print(f"\nAgent chose: RIGHT, Reward = {reward}")

    # Gradient of log pi(right|s) w.r.t. theta
    grad_left = -probs[0]   # = 0 - pi(left)
    grad_right = 1 - probs[1]  # = 1 - pi(right)
    print(f"\ngrad log_pi(right) w.r.t. theta_left  = {grad_left:.3f}")
    print(f"grad log_pi(right) w.r.t. theta_right = {grad_right:.3f}")

    # Update
    theta[0] += alpha * grad_left * reward
    theta[1] += alpha * grad_right * reward
    print(f"\ntheta_left  <- 0 + {alpha}*({grad_left})*{reward} = {theta[0]:.3f}")
    print(f"theta_right <- 0 + {alpha}*({grad_right})*{reward} = {theta[1]:.3f}")

    # New probabilities
    exp_t = np.exp(theta - theta.max())
    new_probs = exp_t / exp_t.sum()
    print(f"\nNew: pi(left) = {new_probs[0]:.3f}, pi(right) = {new_probs[1]:.3f}")
    print(f"[!] RIGHT probability increased from 0.500 to {new_probs[1]:.3f}")


def demo_baseline_effect():
    """Show how baseline changes the advantage signal"""
    print("\n" + "=" * 60)
    print("Demo: Baseline Effect on Advantage")
    print("=" * 60)
    # Simulated returns from CartPole (all positive!)
    returns = np.array([10, 30, 50, 80, 120, 150])
    V = np.array([15, 35, 55, 75, 95, 110])  # estimated value

    print(f"\n{'Step':>5} | {'G_t':>5} | {'V(S_t)':>6} | {'G_t (no bl)':>12} | {'G_t - V':>8}")
    print("-" * 50)
    for i in range(len(returns)):
        raw = returns[i]
        adv = returns[i] - V[i]
        print(f"{i:>5} | {raw:>5} | {V[i]:>6} | {raw:>+12.1f} | {adv:>+8.1f}")

    print(f"\n[!] Without baseline: ALL signals are positive (even bad steps)")
    print(f"    With baseline: advantage can be NEGATIVE (bad steps get penalized)")
    print(f"    Steps where G_t < V(S_t) have negative advantage -> probability decreases")


if __name__ == "__main__":
    print("[TEST] Day 10: Policy Gradient (REINFORCE)\n")

    demo_softmax_gradient()
    demo_baseline_effect()

    print("\n" + "=" * 60)
    print("Training REINFORCE on CartPole")
    print("=" * 60)

    env = SimpleCartPole()

    print("\n--- REINFORCE (no baseline) ---")
    np.random.seed(42)
    r1 = train(env, n_ep=500, baseline=False)

    print("\n--- REINFORCE (with baseline) ---")
    np.random.seed(42)
    r2 = train(env, n_ep=500, baseline=True)

    print(f"\n{'Method':>20} | {'Last50 avg':>10} | {'Best50':>10}")
    print("-" * 48)
    for name, r in [("No baseline", r1), ("With baseline", r2)]:
        last50 = np.mean(r[-50:]); best50 = max(np.mean(r[i:i+50]) for i in range(len(r)-49))
        print(f"{name:>20} | {last50:>10.1f} | {best50:>10.1f}")

    print("\n" + "=" * 60)
    print("[OK] Key takeaways:")
    print("   1. Policy gradient: directly optimize pi_theta(a|s)")
    print("   2. Log-derivative trick: nabla P = P * nabla log P")
    print("   3. REINFORCE: nabla J = E[nabla log pi * G_t], no model needed")
    print("   4. Baseline: subtract V(s) -> no bias, lower variance")
    print("   5. Advantage: A = G_t - V(s), measures how much better than expected")
    print("=" * 60)
