"""
Module 3b: Goal Programming
Optimizes trade-offs between drilling speed, equipment safety, and fuel efficiency.
Minimizes total weighted deviation from ideal targets.
"""
import numpy as np
from scipy.optimize import linprog

# Goal targets (normalised 0–1 scale)
GOALS = {
    "ROP_target":        0.80,   # want ROP at 80 % of max → fast drilling
    "Failure_target":    0.10,   # want failure risk below 10 %
    "Energy_target":     0.70,   # want energy efficiency at 70 %
    "Oil_Potential_target": 0.60, # want oil potential above 60 %
}

# Penalty weights for under-achieving each goal (higher = more important)
WEIGHTS = {
    "ROP":        0.20,
    "Failure":    0.40,   # safety gets highest weight
    "Energy":     0.15,
    "Oil_Potential": 0.25,
}


def goal_programming_score(rop_score, failure_risk, energy_score, oil_score):
    """
    Compute total weighted deviation from ideal goals.
    Returns:
        total_deviation: lower is better (0 = all goals met)
        deviations:      per-goal deviation dict
        recommendation:  text recommendation based on worst deviation
    """
    deviations = {
        "ROP":         max(0, GOALS["ROP_target"]         - rop_score),
        "Failure":     max(0, failure_risk                - GOALS["Failure_target"]),
        "Energy":      max(0, GOALS["Energy_target"]      - energy_score),
        "Oil_Potential": max(0, GOALS["Oil_Potential_target"] - oil_score),
    }

    total = sum(WEIGHTS[k] * v for k, v in deviations.items())

    # Identify worst deviation
    worst = max(deviations, key=lambda k: WEIGHTS[k] * deviations[k])
    recs = {
        "ROP":          "Increase drilling speed — rate of penetration is below target.",
        "Failure":      "⚠️ Equipment failure risk is high — schedule maintenance immediately.",
        "Energy":       "Optimise fuel consumption — energy efficiency is below target.",
        "Oil_Potential":"Reassess reservoir target — oil potential is below expected threshold.",
    }

    return {
        "total_deviation": round(total, 4),
        "deviations":      {k: round(v, 4) for k, v in deviations.items()},
        "worst_goal":      worst,
        "recommendation":  recs[worst],
        "performance_score": round(max(0, 1 - total), 4),
    }


if __name__ == "__main__":
    result = goal_programming_score(0.6, 0.35, 0.5, 0.4)
    print("Goal Programming Result:")
    for k, v in result.items():
        print(f"  {k}: {v}")