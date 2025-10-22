# Methodology (Paper Scaffold)

We formalize dynamic tax design as a Markov decision process where the **state** captures macro shocks, income distribution statistics, and compliance; the **action** is a vector of tax instruments; and the **reward** is social welfare penalized by constraint violations (e.g., revenue adequacy).

Key components:
- Economic simulator with endogenous labor supply and income mobility
- Uncertainty via AR(1) productivity shocks and idiosyncratic noise
- Fairness & distributional metrics (Gini, Atkinson) and deadweight loss
- RL optimization with constraints (Lagrangian shaping and reward clipping)
- Evaluation: off-policy estimators (IS/DR), risk-sensitive (CVaR)
