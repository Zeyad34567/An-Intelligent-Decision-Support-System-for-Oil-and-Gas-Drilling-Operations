"""
Module 2: ML Models
- Formation classifier (drilling sensor logs)
- Oil presence predictor (well log data)
- Equipment failure predictor (AI4I maintenance data)
- Discriminant analysis for well risk tiers
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
#  1. Formation / Lithology Classifier
#     Uses GR, RHOB, NPHI, RDEP, RMED, DTC, PEF
# ─────────────────────────────────────────────
FORMATION_FEATURES = ["GR", "RHOB", "NPHI", "RDEP", "RMED", "DTC", "PEF", "DRHO", "SP"]

def train_formation_model(df):
    """Train a RandomForest to classify geological formations."""
    # Use FORMATION as target; drop Unknown rows
    df_clean = df[df["FORMATION"] != "Unknown"].copy()
    if len(df_clean) < 50:
        return None, None, None

    features = [f for f in FORMATION_FEATURES if f in df_clean.columns]
    X = df_clean[features].fillna(df_clean[features].median())
    y = df_clean["FORMATION"]

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))
    importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)

    return model, le, {"accuracy": acc, "feature_importance": importances, "features": features}


def predict_formation(model, le, row_dict, features):
    """Predict formation for a single depth reading."""
    X = pd.DataFrame([row_dict])[features].fillna(0)
    pred_enc = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    label = le.inverse_transform([pred_enc])[0]
    confidence = proba.max()
    return label, confidence


# ─────────────────────────────────────────────
#  2. Oil Presence Predictor
# ─────────────────────────────────────────────
OIL_FEATURES = [
    "Rock_Type_Code", "Porosity", "Permeability",
    "Trap_Type_Code", "Seismic_Score",
    "Proximity_to_Oil_Field", "Estimated_Reservoir_Depth"
]

def train_oil_model(df):
    """Train GBM to predict oil presence (0/1)."""
    X = df[OIL_FEATURES].fillna(0)
    y = df["Oil_Presence"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_s, y_train)

    acc = accuracy_score(y_test, model.predict(X_test_s))
    importances = pd.Series(model.feature_importances_, index=OIL_FEATURES).sort_values(ascending=False)

    return model, scaler, {"accuracy": acc, "feature_importance": importances}


def predict_oil_presence(model, scaler, input_dict):
    """Predict oil presence probability for a location."""
    X = pd.DataFrame([input_dict])[OIL_FEATURES].fillna(0)
    X_s = scaler.transform(X)
    proba = model.predict_proba(X_s)[0][1]
    prediction = int(proba >= 0.5)
    return prediction, proba


# ─────────────────────────────────────────────
#  3. Equipment Failure Predictor
# ─────────────────────────────────────────────
EQUIP_FEATURES = [
    "Air_Temp_K", "Process_Temp_K", "Rot_Speed_rpm",
    "Torque_Nm", "Tool_Wear_min", "Temp_Diff", "Power", "Type_Code"
]

def train_equipment_model(df):
    """Train RandomForest to predict machine failure."""
    X = df[EQUIP_FEATURES].fillna(0)
    y = df["Machine_Failure"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=150, class_weight="balanced", random_state=42, n_jobs=-1)
    model.fit(X_train_s, y_train)

    acc = accuracy_score(y_test, model.predict(X_test_s))
    importances = pd.Series(model.feature_importances_, index=EQUIP_FEATURES).sort_values(ascending=False)

    return model, scaler, {"accuracy": acc, "feature_importance": importances}


def predict_failure_risk(model, scaler, input_dict):
    """Return failure probability (0–1) for current equipment readings."""
    X = pd.DataFrame([input_dict])[EQUIP_FEATURES].fillna(0)
    X_s = scaler.transform(X)
    proba = model.predict_proba(X_s)[0][1]
    return proba


# ─────────────────────────────────────────────
#  4. Discriminant Analysis — Well Risk Tiers
#     Classifies each depth interval into LOW / MEDIUM / HIGH risk
# ─────────────────────────────────────────────
RISK_FEATURES = ["GR", "RHOB", "NPHI", "RDEP", "ROP"]

def _assign_rule_based_risk(df):
    """Generate rule-based risk labels for LDA training."""
    scores = np.zeros(len(df))
    scores += (df["GR"] > 75).astype(int) * 1          # high gamma-ray → shale risk
    scores += (df["RHOB"] < 2.2).astype(int) * 1        # low density → porous zone
    scores += (df["NPHI"] > 0.3).astype(int) * 1        # high neutron porosity
    scores += (df["RDEP"] < 2.0).astype(int) * 1        # low deep resistivity → wet zone
    if "ROP" in df.columns:
        scores += (df["ROP"] > 80).astype(int) * 1      # fast penetration → soft/unstable

    risk = pd.cut(scores, bins=[-1, 1, 3, 5],
                  labels=["LOW", "MEDIUM", "HIGH"])
    return risk

def train_discriminant_model(df):
    """Train LDA to classify depth intervals by risk tier."""
    features = [f for f in RISK_FEATURES if f in df.columns]
    df_clean = df.dropna(subset=features).copy()
    df_clean["Risk_Label"] = _assign_rule_based_risk(df_clean)
    df_clean = df_clean.dropna(subset=["Risk_Label"])

    X = df_clean[features]
    y = df_clean["Risk_Label"]

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    lda = LinearDiscriminantAnalysis()
    lda.fit(X_s, y)

    acc = accuracy_score(y, lda.predict(X_s))
    return lda, scaler, features, {"accuracy": acc}


def predict_risk_tier(lda, scaler, features, input_dict):
    """Predict risk tier (LOW/MEDIUM/HIGH) for a depth reading."""
    X = pd.DataFrame([input_dict])[features].fillna(0)
    X_s = scaler.transform(X)
    tier = lda.predict(X_s)[0]
    proba = lda.predict_proba(X_s)[0]
    classes = lda.classes_
    proba_dict = dict(zip(classes, proba))
    return tier, proba_dict


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
    from src.data_pipeline import load_drilling_sensor, load_oil_well_log, load_equipment_failure

    print("Training Formation model...")
    df_d = load_drilling_sensor()
    m1, le1, r1 = train_formation_model(df_d)
    if m1:
        print(f"  Accuracy: {r1['accuracy']:.3f}")
        print(f"  Top features: {r1['feature_importance'].head(3).index.tolist()}")

    print("Training Oil Presence model...")
    df_o = load_oil_well_log()
    m2, sc2, r2 = train_oil_model(df_o)
    print(f"  Accuracy: {r2['accuracy']:.3f}")

    print("Training Equipment Failure model...")
    df_e = load_equipment_failure()
    m3, sc3, r3 = train_equipment_model(df_e)
    print(f"  Accuracy: {r3['accuracy']:.3f}")

    print("Training Discriminant (Risk Tier) model...")
    m4, sc4, ft4, r4 = train_discriminant_model(df_d)
    print(f"  Accuracy: {r4['accuracy']:.3f}")

    print("All models trained successfully.")