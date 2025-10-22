# DynTaxRL-PolicyOpt

**Short description**: Research-grade framework for **reinforcement learning of multi-period tax policy** under **macroeconomic uncertainty**, balancing **efficiency–equity trade-offs** with **constraints and governance**. Includes a configurable simulation environment, baseline RL agents, counterfactual evaluation, and reproducible experiments.

## Highlights
- Economic simulator with stochastic productivity & income mobility, labor supply response, and compliance behavior
- Tax instruments: {"τ_labor","τ_capital","VAT","lump_sum_transfers","brackets"} with policy constraints & fairness rules
- Social welfare objectives (utilitarian, Atkinson, Rawlsian), **Gini**/**Atkinson index**, deadweight loss, revenue adequacy
- RL baselines (tabular Q-learning, policy gradient) + policy search
- Off-policy evaluation (IS/DR), ablations, and risk-sensitive objectives (CVaR)
- Reproducible configs, tests, CI, and paper-ready docs

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run a tiny demo (offline, no external APIs)
python -m dtr.cli demo --episodes 30 --out artifacts/demo

# Run tests
pytest -q
```

## Repo Structure
```
DynTaxRL-PolicyOpt/
├─ src/dtr/                   # Core package (env, agents, eval, utils)
├─ configs/                   # Experiment configs
├─ data/synthetic/            # Synthetic priors & distributions
├─ docs/                      # Paper scaffolding & design docs
├─ notebooks/                 # Reproducible exploration
├─ tests/                     # Unit/integration tests
├─ .github/workflows/         # CI
└─ artifacts/                 # Outputs (gitignored)
```
License: MIT © 2025
