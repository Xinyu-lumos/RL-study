"""
Day 11: Actor-Critic (A2C + GAE) Practice

1. A2C with TD error
2. A2C with GAE
3. GAE computation demo
4. Comparison: REINFORCE vs A2C

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


class ActorCritic:
    """Shared-feature Actor-Critic (pure NumPy)"""
    def __init__(self, obs_dim=4, n_act=2, hid=64, lr_a=0.003, lr_c=0.01):
        self.lr_a, self.lr_c, self.n_act = lr_a, lr_c, n_act
        s1, s2 = np.sqrt(2/obs_dim), np.sqrt(2/hid)
        self.W1=np.random.randn(obs_dim,hid)*s1; self.b1=np.zeros(hid)
        self.W2=np.random.randn(hid,hid)*s2;    self.b2=np.zeros(hid)
        self.Wa=np.random.randn(hid,n_act)*np.sqrt(2/hid); self.ba=np.zeros(n_act)
        self.Wv=np.random.randn(hid,1)*np.sqrt(2/hid);     self.bv=np.zeros(1)

    def forward(self, x):
        self.x=np.atleast_2d(x)
        self.h1=np.maximum(0,self.x@self.W1+self.b1)
        self.h2=np.maximum(0,self.h1@self.W2+self.b2)
        self.logits=self.h2@self.Wa+self.ba
        e=np.exp(self.logits-np.max(self.logits,axis=1,keepdims=True))
        self.probs=e/np.sum(e,axis=1,keepdims=True)
        self.value=self.h2@self.Wv+self.bv
        return self.probs, self.value

    def sample(self, state):
        p,_=self.forward(state); return np.random.choice(self.n_act,p=p[0])

    def update(self, states, actions, advs, rets):
        self.forward(np.array(states)); bs=len(states); clip=10.0
        dl=self.probs.copy()
        for i in range(bs): dl[i,actions[i]]-=1; dl[i]*=advs[i]
        dl/=bs
        vf=self.value.flatten(); errs=(vf-np.array(rets)).reshape(-1,1)
        dWa=self.h2.T@dl/bs; dba=np.sum(dl,axis=0)
        dha=dl@self.Wa.T*(self.h2>0)
        dWv=self.h2.T@errs/bs; dbv=np.mean(errs,axis=0)
        dhc=errs@self.Wv.T*(self.h2>0)
        dh2=dha+dhc
        dW2=self.h1.T@dh2/bs; db2=np.mean(dh2,axis=0)
        dh1=dh2@self.W2.T*(self.h1>0)
        dW1=self.x.T@dh1/bs; db1=np.mean(dh1,axis=0)
        for g in[dW1,db1,dW2,db2,dWa,dba,dWv,dbv]: np.clip(g,-clip,clip,out=g)
        self.Wa-=self.lr_a*dWa; self.ba-=self.lr_a*dba
        self.Wv-=self.lr_c*dWv; self.bv-=self.lr_c*dbv
        for g,W,b in[(dW2,self.W2,self.b2),(dW1,self.W1,self.b1)]:
            lr=(self.lr_a+self.lr_c)/2
            W-=lr*g; b-=lr*(db2 if W is self.W2 else db1)


def compute_gae(rewards, values, gamma=0.99, lam=0.95):
    """Compute GAE advantages from rewards and value estimates"""
    advs = []; gae = 0
    for t in reversed(range(len(rewards))):
        nv = 0 if t == len(rewards)-1 else values[t+1]
        delta = rewards[t] + gamma * nv - values[t]
        gae = delta + gamma * lam * gae
        advs.insert(0, gae)
    return np.array(advs)


def train_a2c(env, n_ep=500, gamma=0.99, lam=0.95, lr_a=0.003, lr_c=0.01, use_gae=True):
    model = ActorCritic(lr_a=lr_a, lr_c=lr_c); rewards = []
    for ep in range(n_ep):
        s=env.reset(); S,A,R=[],[],[]; done=False
        while not done:
            a=model.sample(s); ns,r,done=env.step(a)
            S.append(s); A.append(a); R.append(r); s=ns
        _, vals = model.forward(np.array(S)); vals=vals.flatten()
        if use_gae:
            advs = compute_gae(R, vals, gamma, lam)
            rets = advs + vals
        else:
            # Simple TD(0) advantage
            advs = []; rets = []
            for t in range(len(R)):
                nv = 0 if t==len(R)-1 else R[t+1]+gamma*vals[t+1] if t<len(R)-1 else 0
                # recalculate: td target
                if t < len(R)-1:
                    td_target = R[t] + gamma * vals[t+1]
                else:
                    td_target = R[t]
                advs.append(td_target - vals[t])
                rets.append(td_target)
            advs = np.array(advs); rets = np.array(rets)
        if len(advs)>1: advs=(advs-advs.mean())/(advs.std()+1e-8)
        model.update(S,A,advs,rets)
        rewards.append(sum(R))
        if(ep+1)%100==0: print(f"  Ep {ep+1:>4}: avg={np.mean(rewards[-100:]):>6.1f}")
    return rewards


def demo_gae():
    """GAE computation step-by-step"""
    print("=" * 60)
    print("Demo: GAE Computation Step-by-Step")
    print("=" * 60)
    gamma, lam = 0.99, 0.95
    rewards = [1.0, -0.5, 10.0]
    values  = [2.0,  1.5,  9.0]

    # Compute TD residuals
    deltas = []
    for t in range(len(rewards)):
        nv = 0 if t==len(rewards)-1 else values[t+1]
        d = rewards[t] + gamma*nv - values[t]
        deltas.append(d)

    print(f"\ngamma={gamma}, lambda={lam}")
    print(f"\n{'t':>3} | {'R':>6} | {'V':>6} | {'delta':>8}")
    print("-" * 32)
    for t in range(len(rewards)):
        print(f"{t:>3} | {rewards[t]:>6.1f} | {values[t]:>6.1f} | {deltas[t]:>8.3f}")

    # Compute GAE for different lambda values
    print(f"\n{'t':>3} | {'GAE(0)':>8} | {'GAE(0.5)':>10} | {'GAE(0.95)':>10} | {'GAE(1.0)':>10}")
    print("-" * 50)

    for lam_val in [0.0, 0.5, 0.95, 1.0]:
        advs = []; gae = 0
        for t in reversed(range(len(rewards))):
            gae = deltas[t] + gamma * lam_val * gae
            advs.insert(0, gae)
        if lam_val == 0.0: col0 = [f"{a:>8.3f}" for a in advs]
        elif lam_val == 0.5: col1 = [f"{a:>10.3f}" for a in advs]
        elif lam_val == 0.95: col2 = [f"{a:>10.3f}" for a in advs]
        else: col3 = [f"{a:>10.3f}" for a in advs]

    for t in range(len(rewards)):
        print(f"{t:>3} | {col0[t]} | {col1[t]} | {col2[t]} | {col3[t]}")

    print(f"\n[!] lambda=0 -> TD(0) advantages (low variance, high bias)")
    print(f"    lambda=1 -> MC advantages (high variance, unbiased)")
    print(f"    lambda=0.95 -> balanced (recommended)")


if __name__ == "__main__":
    print("[TEST] Day 11: Actor-Critic (A2C + GAE)\n")

    demo_gae()

    env = SimpleCartPole()

    print("\n" + "=" * 60)
    print("Training A2C on CartPole")
    print("=" * 60)

    print("\n--- A2C with GAE (lambda=0.95) ---")
    np.random.seed(42)
    r1 = train_a2c(env, n_ep=500, use_gae=True, lam=0.95)

    print("\n--- A2C with TD(0) (lambda=0) ---")
    np.random.seed(42)
    r2 = train_a2c(env, n_ep=500, use_gae=True, lam=0.0)

    print(f"\n{'Method':>20} | {'Last50':>8} | {'Best50':>8}")
    print("-" * 42)
    for name, r in [("A2C+GAE(0.95)", r1), ("A2C+TD(0)", r2)]:
        last = np.mean(r[-50:]); best = max(np.mean(r[i:i+50]) for i in range(len(r)-49))
        print(f"{name:>20} | {last:>8.1f} | {best:>8.1f}")

    print("\n" + "=" * 60)
    print("[OK] Key takeaways:")
    print("   1. Actor-Critic: Actor selects + Critic evaluates, update every step")
    print("   2. TD error as advantage: delta = R + gamma*V(S') - V(S), low variance")
    print("   3. A2C: synchronous batch update, simple and effective")
    print("   4. GAE: A_t = delta_t + gamma*lambda*A_{t+1}, tunable bias-variance")
    print("   5. lambda=0->TD(0), lambda=1->MC, lambda=0.95->recommended")
    print("=" * 60)
