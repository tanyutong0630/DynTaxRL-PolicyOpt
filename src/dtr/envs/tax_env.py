import numpy as np

class TaxEnv:
    """Stylized macro-micro tax environment.

    State (vector): [shock_level, gini, revenue_ratio, compliance, mean_income]
    Action (vector): [tau_labor, tau_capital, vat, transfer] within bounds
    Reward: social welfare - penalties for constraints (e.g., revenue floor)
    """
    def __init__(self, cfg):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.get("seed", 1337))
        self.pop = cfg["env"]["population_size"]
        self.income_grid = np.array(cfg["env"]["income_grid"], dtype=float)
        self.bounds = cfg["tax_instruments"]
        self.reset()

    def reset(self):
        self.t = 0
        self.shock = 1.0
        self.state = np.array([0.0, 0.35, 0.18, self.cfg["env"]["compliance_base"], np.mean(self.income_grid)])
        return self.state.copy()

    def _clip_action(self, a):
        # map [0,1]^4 to instrument bounds
        keys = ["tau_labor","tau_capital","vat","transfer"]
        out = np.zeros(4)
        for i,k in enumerate(keys):
            lo, hi = self.bounds[k]
            out[i] = lo + (hi - lo) * np.clip(a[i], 0.0, 1.0)
        return out

    def step(self, a):
        a = self._clip_action(np.asarray(a))
        # AR(1) macro shock on productivity
        rho = self.cfg["env"]["shock_rho"]
        sigma = self.cfg["env"]["shock_sigma"]
        eps = self.rng.normal(0, sigma)
        self.shock = rho * self.shock + eps
        prod = np.exp(self.shock)

        # Simulate incomes & labor response to tau_labor
        labor_el = self.cfg["env"]["labor_elasticity"]
        base = self.income_grid * prod
        labor_scale = (1 - a[0]) ** labor_el  # higher tax lowers labor supply
        incomes = base * labor_scale

        # Revenue from labor, capital (proxy with upper grid), VAT; simple compliance
        comp = max(0.5, self.state[3] - 0.1 * a[0])  # lower with higher tax
        revenue = (a[0] * np.mean(incomes) + a[1] * np.mean(self.income_grid[-2:]) + a[2] * np.mean(incomes)) * comp
        transfer = a[3]

        mean_income = float(np.mean(incomes))
        gini = self._gini(incomes - transfer)  # naive redistribution
        revenue_ratio = float(revenue / max(1e-6, mean_income))

        # Welfare (Atkinson / utilitarian / Rawlsian)
        welfare = self._welfare(incomes, transfer)

        # Penalties: revenue floor
        rev_floor = self.cfg["objective"]["revenue_floor"]
        penalty = 0.0
        if revenue_ratio < rev_floor:
            penalty += (rev_floor - revenue_ratio) * 10.0

        reward = welfare - penalty

        self.state = np.array([self.shock, gini, revenue_ratio, comp, mean_income])
        done = (self.t >= self.cfg["horizon"]) or (np.isnan(reward))
        self.t += 1
        info = {"welfare": welfare, "penalty": penalty, "gini": gini, "revenue_ratio": revenue_ratio}
        return self.state.copy(), float(reward), done, info

    def _welfare(self, incomes, transfer):
        rule = self.cfg["objective"]["welfare"]
        if rule == "utilitarian":
            return float(np.mean(np.log(1 + incomes + transfer)))
        elif rule == "rawls":
            return float(np.min(np.log(1 + incomes + transfer)))
        else:  # Atkinson
            eps = self.cfg["objective"]["epsilon"]
            y = incomes + transfer
            y = np.maximum(y, 1e-6)
            if abs(1-eps) < 1e-6:
                return float(np.mean(np.log(y)))
            return float((np.mean(y**(1-eps)))**(1/(1-eps)))

    def _gini(self, x):
        x = np.sort(np.maximum(x, 0))
        n = len(x)
        if n == 0:
            return 0.0
        cumx = np.cumsum(x)
        return 1 - (2/(n-1)) * (n - (cumx.sum()/cumx[-1])) if cumx[-1] > 0 else 0.0
