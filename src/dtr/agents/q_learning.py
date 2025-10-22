import numpy as np

class QLearningAgent:
    def __init__(self, env, cfg):
        self.env = env
        self.cfg = cfg
        self.alpha = cfg["agent"]["alpha"]
        self.gamma = cfg["agent"]["gamma"]
        self.eps = cfg["agent"]["epsilon"]
        # Discretize state & action for tabular Q
        self.state_bins = [np.linspace(-1.5,1.5,7), np.linspace(0,1,6), np.linspace(0,1,6), np.linspace(0.5,1,6), np.linspace(0,1e5,6)]
        self.action_bins = [np.linspace(0,1,6) for _ in range(4)]
        self.Q = np.zeros([*(len(b)-1 for b in self.state_bins), *(len(b)-1 for b in self.action_bins)])

    def _digitize(self, s, bins):
        return tuple(np.digitize(v, b)-1 for v,b in zip(s, bins))

    def _sample_action(self):
        idxs = [np.random.randint(len(b)-1) for b in self.action_bins]
        return np.array([ (self.action_bins[i][k]+self.action_bins[i][k+1])/2 for i,k in enumerate(idxs) ])

    def act(self, s):
        s_i = self._digitize(s, self.state_bins)
        if np.random.rand() < self.eps:
            return self._sample_action()
        q_slice = self.Q[s_i]
        # greedy over discretized actions
        a_i = np.unravel_index(np.argmax(q_slice), q_slice.shape)
        return np.array([ (self.action_bins[i][k]+self.action_bins[i][k+1])/2 for i,k in enumerate(a_i) ])

    def observe(self, s, a, r, s2, done):
        s_i = self._digitize(s, self.state_bins)
        a_i = tuple(np.digitize(a[i], self.action_bins[i])-1 for i in range(4))
        s2_i = self._digitize(s2, self.state_bins)
        target = r + (0 if done else self.gamma * np.max(self.Q[s2_i]))
        self.Q[s_i + a_i] = (1 - self.alpha) * self.Q[s_i + a_i] + self.alpha * target

    def end_episode(self):
        pass
