from typer import Typer, Option
from rich import print
from pathlib import Path
from .utils.config import load_yaml
from .envs.tax_env import TaxEnv
from .agents.q_learning import QLearningAgent
from .agents.pg import PolicyGradientAgent
from .eval.metrics import compute_metrics_summary

app = Typer(help="DynTaxRL Policy Optimization CLI")

@app.command()
def demo(episodes: int = Option(50, help="Training episodes"),
         out: str = Option("artifacts/demo", help="Output directory"),
         config: str = Option("configs/experiment.yaml", help="Config path")):
    cfg = load_yaml(config)
    env = TaxEnv(cfg)
    algo = cfg["agent"]["algo"]
    if algo == "q_learning":
        agent = QLearningAgent(env, cfg)
    else:
        agent = PolicyGradientAgent(env, cfg)

    Path(out).mkdir(parents=True, exist_ok=True)
    returns = []
    for ep in range(episodes):
        s = env.reset()
        done = False
        G = 0.0
        while not done:
            a = agent.act(s)
            s2, r, done, info = env.step(a)
            agent.observe(s, a, r, s2, done)
            s = s2
            G += r
        agent.end_episode()
        returns.append(G)
        if (ep+1) % max(1, cfg["logging"]["every"]) == 0:
            print(f"[cyan]Episode {ep+1}[/cyan] return: {G:.3f}")

    # final policy evaluation
    summary = compute_metrics_summary(env, policy=agent, episodes=10)
    (Path(out) / "summary.json").write_text(__import__("json").dumps(summary, indent=2))
    print(f"[bold green]Done.[/bold green] Artifacts in {out}")

if __name__ == "__main__":
    app()
