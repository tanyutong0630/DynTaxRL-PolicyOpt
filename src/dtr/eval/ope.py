# Placeholders for Off-Policy Evaluation: IS / DR
def importance_sampling(*args, **kwargs):
    return {"estimator": "IS", "value": None}

def doubly_robust(*args, **kwargs):
    return {"estimator": "DR", "value": None}
