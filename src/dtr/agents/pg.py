import numpy as np

class PolicyGradientAgent:
    def __init__(self, env, cfg):
        self.env = env
        self.cfg = cfg
        self.lr = cfg["agent"]["alpha"]
        self.gamma = cfg["agent"]["gamma"]
        self.theta = np.zeros(5*4).reshape(5,4)  # simple linear policy: a = sigmoid(s @ theta)
        self.traj = []

    def _sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def act(self, s):
        logits = s @ self.theta
        a = self._sigmoid(logits)  # 0..1 per action dim
        self.traj.append({"s": s.copy(), "a": a.copy()})
        return a

    def observe(self, s, a, r, s2, done):
        self.traj[-1]["r"] = r

    def end_episode(self):
        # REINFORCE with baseline (mean return)
        Gs, ret = [], 0.0
        for step in reversed(self.traj):
            ret = step["r"] + self.gamma * ret
            Gs.append(ret)
        Gs = list(reversed(Gs))
        baseline = np.mean(Gs)
        for step, G in zip(self.traj, Gs):
            s = step["s"]; a = step["a"]
            grad = np.outer(s, a*(1-a))  # d(sigmoid)/d(theta)
            self.theta += self.lr * (G - baseline) * grad
        self.traj = []
