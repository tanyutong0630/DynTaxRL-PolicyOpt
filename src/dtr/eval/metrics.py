import numpy as np, json

def rollout(env, policy, episodes=5):
    outs = []
    for _ in range(episodes):
        s = env.reset()
        done = False
        ret = 0.0
        while not done:
            a = policy.act(s)
            s, r, done, info = env.step(a)
            ret += r
        outs.append({"return": ret, **info})
        policy.end_episode()
    return outs

def compute_metrics_summary(env, policy, episodes=5):
    outs = rollout(env, policy, episodes)
    def mean(key): return float(np.mean([o[key] for o in outs]))
    summary = {
        "episodes": episodes,
        "avg_return": mean("return"),
        "avg_welfare": mean("welfare"),
        "avg_gini": mean("gini"),
        "avg_revenue_ratio": mean("revenue_ratio")
    }
    return summary
