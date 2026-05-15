"""
Module: Decision Analysis
Implements 4 classic IDSS decision-making methods:

1. Discriminant Analysis      — classifies wells into risk tiers (LOW/MEDIUM/HIGH)
2. Decision Under Certainty   — picks the best alternative when outcomes are known
3. Decision Under Risk        — uses EMV/expected value with known probabilities
4. Decision Under Uncertainty — Maximin, Maximax, Hurwicz, Minimax Regret, Laplace
"""

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix


# ══════════════════════════════════════════════════════
#  1. DISCRIMINANT ANALYSIS — Well Risk Classification
# ══════════════════════════════════════════════════════

DA_FEATURES = ["GR", "RHOB", "NPHI", "RDEP", "ROP", "MUDWEIGHT", "DTC", "PEF"]

def _build_risk_labels(df: pd.DataFrame) -> pd.Series:
    """
    Rule-based risk scoring to generate training labels for LDA.
    Each condition adds to a cumulative risk score (0–8).
    """
    score = pd.Series(np.zeros(len(df)), index=df.index)

    if "GR" in df.columns:
        score += (df["GR"] > 75).astype(int)          # shale indicator
        score += (df["GR"] > 100).astype(int)          # severe shale

    if "RHOB" in df.columns:
        score += (df["RHOB"] < 2.2).astype(int)        # under-compacted zone
        score += (df["RHOB"] < 2.0).astype(int)        # severe

    if "NPHI" in df.columns:
        score += (df["NPHI"] > 0.30).astype(int)       # high fluid content

    if "RDEP" in df.columns:
        score += (df["RDEP"] < 2.0).astype(int)        # conductive / wet zone

    if "ROP" in df.columns:
        score += (df["ROP"] > 90).astype(int)          # fast = soft/unstable

    if "MUDWEIGHT" in df.columns:
        score += (df["MUDWEIGHT"] > 1.5).astype(int)   # heavy mud = pressure concern

    labels = pd.cut(
        score,
        bins=[-1, 2, 4, 8],
        labels=["LOW", "MEDIUM", "HIGH"]
    )
    return labels


def train_discriminant_analysis(df: pd.DataFrame):
    """
    Train Linear Discriminant Analysis on well log features.
    Returns: model, scaler, feature list, metrics dict
    """
    features = [f for f in DA_FEATURES if f in df.columns]
    df_clean = df[features].copy()
    df_clean["Risk_Label"] = _build_risk_labels(df)
    df_clean = df_clean.dropna()

    X = df_clean[features]
    y = df_clean["Risk_Label"]

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    lda = LinearDiscriminantAnalysis(solver="svd")
    lda.fit(X_s, y)

    y_pred = lda.predict(X_s)
    report = classification_report(y, y_pred, output_dict=True)
    cm     = confusion_matrix(y, y_pred, labels=["LOW","MEDIUM","HIGH"])

    # LDA coefficients — which features discriminate most
    coeff_df = pd.DataFrame(
        lda.coef_,
        columns=features,
        index=lda.classes_
    )

    return lda, scaler, features, {
        "accuracy":    report["accuracy"],
        "report":      report,
        "confusion_matrix": cm,
        "classes":     list(lda.classes_),
        "coefficients": coeff_df,
        "features":    features,
    }


def predict_risk_tier(lda, scaler, features, reading: dict):
    """
    Predict risk tier for a single depth reading.
    Returns tier label and probability dict.
    """
    X = pd.DataFrame([reading])[features].fillna(0)
    X_s = scaler.transform(X)
    tier  = lda.predict(X_s)[0]
    proba = lda.predict_proba(X_s)[0]
    return tier, dict(zip(lda.classes_, proba))


def classify_all_depths(lda, scaler, features, df: pd.DataFrame) -> pd.Series:
    """Apply LDA risk classification to every row in df."""
    avail = [f for f in features if f in df.columns]
    X = df[avail].fillna(0)
    X_s = scaler.transform(X)
    return pd.Series(lda.predict(X_s), index=df.index)


# ══════════════════════════════════════════════════════
#  2. DECISION UNDER CERTAINTY
#     All outcomes are known → pick the best alternative
# ══════════════════════════════════════════════════════

DRILLING_ALTERNATIVES = {
    "Continue Drilling":         {"description": "Keep drilling at current parameters"},
    "Adjust Mud Weight":         {"description": "Modify mud density to stabilise wellbore"},
    "Reduce ROP":                {"description": "Slow penetration rate to reduce risk"},
    "Pull Out & Inspect":        {"description": "POOH for bit/BHA inspection"},
    "Sidetrack":                 {"description": "Drill new wellbore section avoiding hazard"},
}

def decision_under_certainty(scores: dict) -> dict:
    """
    Decision under certainty: outcomes fully known.
    scores: {alternative_name: utility_value (0–1)}
    Returns ranked alternatives and best choice.
    """
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best   = ranked[0]
    df_out = pd.DataFrame(ranked, columns=["Alternative", "Utility_Score"])
    df_out["Rank"] = range(1, len(df_out) + 1)
    df_out["Recommendation"] = df_out["Rank"].apply(
        lambda r: "✅ BEST CHOICE" if r == 1 else ("⚠️ ACCEPTABLE" if r == 2 else "❌ NOT RECOMMENDED")
    )
    return {
        "best_alternative": best[0],
        "best_score":       round(best[1], 4),
        "ranked_table":     df_out,
        "explanation":      f"Under certainty, '{best[0]}' yields the highest utility ({best[1]:.1%}). "
                            f"All outcomes are deterministic — no probability weighting needed.",
    }


def compute_certainty_scores(risk_tier: str, failure_prob: float,
                              oil_prob: float, rop_score: float) -> dict:
    """
    Compute utility scores for each drilling alternative given current conditions.
    Higher score = better outcome for that alternative.
    """
    risk_num = {"LOW": 0.1, "MEDIUM": 0.5, "HIGH": 0.9}.get(risk_tier, 0.5)

    return {
        "Continue Drilling":  round((1 - risk_num) * 0.4 + rop_score * 0.4 + oil_prob * 0.2, 4),
        "Adjust Mud Weight":  round((1 - risk_num) * 0.5 + 0.3 + oil_prob * 0.2, 4),
        "Reduce ROP":         round((1 - failure_prob) * 0.5 + (1 - risk_num) * 0.3 + 0.2, 4),
        "Pull Out & Inspect": round((1 - failure_prob) * 0.6 + 0.2 + (1 - risk_num) * 0.2, 4),
        "Sidetrack":          round(oil_prob * 0.5 + (1 - risk_num) * 0.3 + 0.2, 4),
    }


# ══════════════════════════════════════════════════════
#  3. DECISION UNDER RISK
#     Probabilities are known → use Expected Monetary Value
# ══════════════════════════════════════════════════════

# Payoff table: alternative × scenario → value (normalised utility, higher = better)
# Scenarios: Stable Formation, Minor Hazard, Major Hazard, Equipment Failure
PAYOFF_TABLE = {
    #                          Stable   Minor   Major   Equip
    "Continue Drilling":      [ 0.95,   0.60,  -0.40,  -0.30],
    "Adjust Mud Weight":      [ 0.70,   0.75,   0.30,   0.10],
    "Reduce ROP":             [ 0.65,   0.70,   0.50,   0.40],
    "Pull Out & Inspect":     [ 0.40,   0.50,   0.70,   0.80],
    "Sidetrack":              [ 0.50,   0.55,   0.60,   0.30],
}
SCENARIOS = ["Stable Formation", "Minor Hazard", "Major Hazard", "Equipment Failure"]


def decision_under_risk(probabilities: list) -> dict:
    """
    Decision under risk using Expected Monetary Value (EMV).
    probabilities: [P(Stable), P(Minor), P(Major), P(Equipment Failure)] — must sum to 1.
    Returns EMV for each alternative, best choice, and full payoff matrix.
    """
    probs = np.array(probabilities)
    probs = probs / probs.sum()   # normalise to ensure sum=1

    emv_results = {}
    for alt, payoffs in PAYOFF_TABLE.items():
        emv = float(np.dot(probs, payoffs))
        emv_results[alt] = round(emv, 4)

    ranked  = sorted(emv_results.items(), key=lambda x: x[1], reverse=True)
    best    = ranked[0]

    # Full payoff dataframe
    df_payoff = pd.DataFrame(PAYOFF_TABLE, index=SCENARIOS).T
    df_payoff.columns = SCENARIOS
    df_payoff["EMV"]  = [emv_results[a] for a in df_payoff.index]
    df_payoff["Rank"] = df_payoff["EMV"].rank(ascending=False).astype(int)

    return {
        "best_alternative": best[0],
        "best_emv":         best[1],
        "emv_scores":       emv_results,
        "payoff_table":     df_payoff,
        "probabilities":    dict(zip(SCENARIOS, probs.tolist())),
        "explanation":      f"Using EMV with given scenario probabilities, "
                            f"'{best[0]}' yields the highest expected value ({best[1]:.4f}). "
                            f"EMV = Σ (probability × payoff) across all scenarios.",
    }


def estimate_scenario_probabilities(risk_tier: str, failure_prob: float) -> list:
    """
    Estimate scenario probabilities from current DSS readings.
    Returns [P(Stable), P(Minor), P(Major), P(Equipment_Failure)]
    """
    risk_num = {"LOW": 0.1, "MEDIUM": 0.4, "HIGH": 0.75}.get(risk_tier, 0.4)
    p_stable  = max(0.05, 1.0 - risk_num - failure_prob * 0.5)
    p_minor   = min(0.40, risk_num * 0.5)
    p_major   = min(0.35, risk_num * 0.5)
    p_equip   = min(0.30, failure_prob)
    total     = p_stable + p_minor + p_major + p_equip
    return [round(p / total, 4) for p in [p_stable, p_minor, p_major, p_equip]]


# ══════════════════════════════════════════════════════
#  4. DECISION UNDER UNCERTAINTY
#     Probabilities unknown → 5 classic criteria
# ══════════════════════════════════════════════════════

def decision_under_uncertainty(alpha: float = 0.5) -> dict:
    """
    Apply 5 uncertainty criteria to the PAYOFF_TABLE.
    alpha: Hurwicz optimism coefficient (0=pessimist, 1=optimist)

    Returns results for:
    - Maximin  (Wald)      — pessimistic: maximise the minimum payoff
    - Maximax             — optimistic:  maximise the maximum payoff
    - Hurwicz             — weighted mix of best and worst
    - Minimax Regret      — minimise maximum opportunity loss
    - Laplace (Bayes)     — assume equal probabilities, pick highest average
    """
    alts    = list(PAYOFF_TABLE.keys())
    matrix  = np.array([PAYOFF_TABLE[a] for a in alts])  # shape (5 alts, 4 scenarios)

    # 1. Maximin: best of the worst-case payoffs
    row_mins   = matrix.min(axis=1)
    maximin_idx = int(np.argmax(row_mins))

    # 2. Maximax: best of the best-case payoffs
    row_maxes   = matrix.max(axis=1)
    maximax_idx = int(np.argmax(row_maxes))

    # 3. Hurwicz: alpha * max + (1-alpha) * min
    hurwicz_scores = alpha * row_maxes + (1 - alpha) * row_mins
    hurwicz_idx    = int(np.argmax(hurwicz_scores))

    # 4. Minimax Regret: build regret matrix, minimise max regret
    col_maxes     = matrix.max(axis=0)          # best payoff per scenario
    regret_matrix = col_maxes - matrix          # opportunity loss
    row_max_regret  = regret_matrix.max(axis=1)
    minimax_idx   = int(np.argmin(row_max_regret))

    # 5. Laplace: equal probability across scenarios
    laplace_scores = matrix.mean(axis=1)
    laplace_idx    = int(np.argmax(laplace_scores))

    # Build summary dataframe
    df_summary = pd.DataFrame({
        "Alternative":   alts,
        "Maximin":       row_mins.round(3),
        "Maximax":       row_maxes.round(3),
        "Hurwicz":       hurwicz_scores.round(3),
        "Max_Regret":    row_max_regret.round(3),
        "Laplace_Avg":   laplace_scores.round(3),
    })

    results = {
        "Maximin":       alts[maximin_idx],
        "Maximax":       alts[maximax_idx],
        "Hurwicz":       alts[hurwicz_idx],
        "Minimax_Regret":alts[minimax_idx],
        "Laplace":       alts[laplace_idx],
    }

    # Consensus: most frequently recommended alternative
    from collections import Counter
    votes     = Counter(results.values())
    consensus = votes.most_common(1)[0][0]

    # Regret matrix as dataframe
    df_regret = pd.DataFrame(regret_matrix, index=alts, columns=SCENARIOS).round(3)

    return {
        "criteria_winners": results,
        "consensus":        consensus,
        "votes":            dict(votes),
        "summary_table":    df_summary,
        "regret_matrix":    df_regret,
        "alpha":            alpha,
        "explanations": {
            "Maximin":        "Pessimistic (Wald): Choose the alternative with the best worst-case outcome.",
            "Maximax":        "Optimistic: Choose the alternative with the best best-case outcome.",
            "Hurwicz":        f"Balanced (α={alpha}): Weighted mix of best and worst — α controls optimism.",
            "Minimax_Regret": "Regret-based: Minimise the maximum opportunity loss vs the best possible payoff.",
            "Laplace":        "Equal probability: Assume all scenarios are equally likely; pick highest average.",
        },
    }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
    from src.data_pipeline import load_drilling_sensor

    print("=== 1. DISCRIMINANT ANALYSIS ===")
    df = load_drilling_sensor()
    lda, scaler, feats, metrics = train_discriminant_analysis(df)
    print(f"  Accuracy: {metrics['accuracy']:.1%}")
    print(f"  Classes:  {metrics['classes']}")

    print("\n=== 2. DECISION UNDER CERTAINTY ===")
    scores = compute_certainty_scores("HIGH", 0.4, 0.6, 0.5)
    result = decision_under_certainty(scores)
    print(f"  Best: {result['best_alternative']}  ({result['best_score']:.3f})")

    print("\n=== 3. DECISION UNDER RISK ===")
    probs  = estimate_scenario_probabilities("HIGH", 0.4)
    result = decision_under_risk(probs)
    print(f"  Best EMV: {result['best_alternative']}  ({result['best_emv']:.4f})")
    print(f"  Probabilities: {result['probabilities']}")

    print("\n=== 4. DECISION UNDER UNCERTAINTY ===")
    result = decision_under_uncertainty(alpha=0.5)
    print(f"  Maximin:        {result['criteria_winners']['Maximin']}")
    print(f"  Maximax:        {result['criteria_winners']['Maximax']}")
    print(f"  Hurwicz:        {result['criteria_winners']['Hurwicz']}")
    print(f"  Minimax Regret: {result['criteria_winners']['Minimax_Regret']}")
    print(f"  Laplace:        {result['criteria_winners']['Laplace']}")
    print(f"  Consensus:      {result['consensus']}")