"""
Module: AHP — Analytic Hierarchy Process for Risk Weight Selection
Weights risk criteria to support drilling decisions.

Criteria:
  - Geological_Risk      (formation instability, lithology uncertainty)
  - Equipment_Risk       (failure probability, tool wear)
  - Oil_Potential        (reservoir quality, oil presence probability)
  - Energy_Efficiency    (fuel consumption, operational cost)
"""

import numpy as np
import pandas as pd

CRITERIA = ["Geological_Risk", "Equipment_Risk", "Oil_Potential", "Energy_Efficiency"]

# Saaty scale: 1=equal, 3=moderate, 5=strong, 7=very strong, 9=extreme
# Matrix: row criterion vs column criterion
# Oil_Potential is most important (primary objective)
# Geological_Risk second (safety), Equipment_Risk third, Energy_Efficiency least
DEFAULT_PAIRWISE = np.array([
    #  Geo    Equip   Oil    Energy
    [  1.0,   2.0,   1/3,   3.0  ],   # Geological_Risk
    [  1/2,   1.0,   1/4,   2.0  ],   # Equipment_Risk
    [  3.0,   4.0,   1.0,   5.0  ],   # Oil_Potential
    [  1/3,   1/2,   1/5,   1.0  ],   # Energy_Efficiency
], dtype=float)

SAATY_RI = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41}

SAATY_SCALE = {
    1: "Equal importance",
    2: "Weak",
    3: "Moderate importance",
    4: "Moderate plus",
    5: "Strong importance",
    6: "Strong plus",
    7: "Very strong importance",
    8: "Very very strong",
    9: "Extreme importance",
}


def compute_ahp_weights(matrix: np.ndarray = None) -> dict:
    """
    Full AHP calculation from a pairwise comparison matrix.
    Returns weights, lambda_max, CI, CR, and consistency verdict.
    """
    if matrix is None:
        matrix = DEFAULT_PAIRWISE
    n = matrix.shape[0]

    # Step 1: Normalise each column
    col_sums = matrix.sum(axis=0)
    norm     = matrix / col_sums

    # Step 2: Priority vector = row means
    weights  = norm.mean(axis=1)

    # Step 3: Consistency
    weighted_sum = matrix @ weights
    lambda_max   = float(np.mean(weighted_sum / weights))
    CI           = (lambda_max - n) / (n - 1)
    CR           = CI / SAATY_RI.get(n, 1.49)

    weight_dict = dict(zip(CRITERIA, weights))

    return {
        "weights":     weight_dict,
        "lambda_max":  round(lambda_max, 4),
        "CI":          round(CI, 4),
        "CR":          round(CR, 4),
        "consistent":  CR < 0.10,
        "norm_matrix": pd.DataFrame(norm, index=CRITERIA, columns=CRITERIA).round(4),
        "weight_series": pd.Series(weights, index=CRITERIA).sort_values(ascending=False),
    }


def build_pairwise_from_sliders(geo_vs_equip, geo_vs_oil, geo_vs_energy,
                                 equip_vs_oil, equip_vs_energy, oil_vs_energy) -> np.ndarray:
    """
    Build a 4x4 pairwise matrix from 6 user-provided comparison values.
    Input values follow Saaty scale (1–9); reciprocals filled automatically.
    """
    m = np.ones((4, 4))
    pairs = [
        (0, 1, geo_vs_equip),
        (0, 2, geo_vs_oil),
        (0, 3, geo_vs_energy),
        (1, 2, equip_vs_oil),
        (1, 3, equip_vs_energy),
        (2, 3, oil_vs_energy),
    ]
    for i, j, v in pairs:
        m[i][j] = v
        m[j][i] = 1.0 / v
    return m


def compute_ahp_score(geological_risk: float, equipment_risk: float,
                       oil_potential: float, energy_efficiency: float,
                       matrix: np.ndarray = None) -> tuple:
    """
    Compute a single composite weighted score (0–1).
    All inputs on 0–1 scale where 1 = best.
    Returns (composite_score, ahp_result_dict).
    """
    result = compute_ahp_weights(matrix)
    w      = result["weights"]
    score  = (
        w["Geological_Risk"]   * geological_risk   +
        w["Equipment_Risk"]    * equipment_risk    +
        w["Oil_Potential"]     * oil_potential     +
        w["Energy_Efficiency"] * energy_efficiency
    )
    return round(float(score), 4), result


def sensitivity_analysis(base_matrix: np.ndarray = None) -> pd.DataFrame:
    """
    Run AHP sensitivity: vary each pairwise judgment by ±1 Saaty unit
    and show how weights change. Returns a dataframe of scenarios.
    """
    if base_matrix is None:
        base_matrix = DEFAULT_PAIRWISE

    base = compute_ahp_weights(base_matrix)
    rows = [{"Scenario": "Base Case", **{k: round(v, 4) for k, v in base["weights"].items()},
             "CR": base["CR"]}]

    # Vary Oil_Potential importance (row/col 2) — most impactful criterion
    for delta, label in [(-1, "Oil less important (÷2)"), (+1, "Oil more important (×2)")]:
        m = base_matrix.copy()
        for j in range(4):
            if j != 2:
                new_val = max(1/9, min(9, m[2][j] + delta))
                m[2][j] = new_val
                m[j][2] = 1.0 / new_val
        r = compute_ahp_weights(m)
        rows.append({"Scenario": label, **{k: round(v, 4) for k, v in r["weights"].items()},
                     "CR": r["CR"]})

    for delta, label in [(-1, "Equipment less important (÷2)"), (+1, "Equipment more important (×2)")]:
        m = base_matrix.copy()
        for j in range(4):
            if j != 1:
                new_val = max(1/9, min(9, m[1][j] + delta))
                m[1][j] = new_val
                m[j][1] = 1.0 / new_val
        r = compute_ahp_weights(m)
        rows.append({"Scenario": label, **{k: round(v, 4) for k, v in r["weights"].items()},
                     "CR": r["CR"]})

    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("=== AHP Risk Weight Selection ===")
    r = compute_ahp_weights()
    print("\nPriority Weights:")
    for k, v in sorted(r["weights"].items(), key=lambda x: -x[1]):
        print(f"  {k:25s}: {v:.4f}  ({v:.1%})")
    print(f"\nλ_max = {r['lambda_max']}  |  CI = {r['CI']}  |  CR = {r['CR']}")
    print(f"Consistency: {'✅ CONSISTENT (CR < 0.10)' if r['consistent'] else '❌ INCONSISTENT'}")

    print("\n=== Sensitivity Analysis ===")
    sa = sensitivity_analysis()
    print(sa.to_string(index=False))

    print("\n=== Custom Pairwise (interactive example) ===")
    m2 = build_pairwise_from_sliders(2, 1/4, 3, 1/5, 2, 4)
    r2 = compute_ahp_weights(m2)
    print(f"  CR = {r2['CR']:.4f} — {'Consistent' if r2['consistent'] else 'INCONSISTENT'}")
    for k, v in sorted(r2["weights"].items(), key=lambda x: -x[1]):
        print(f"  {k}: {v:.4f}")