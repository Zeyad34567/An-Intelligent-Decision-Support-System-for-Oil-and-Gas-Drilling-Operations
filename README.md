# Drilling Intelligence Decision Support System
### An Intelligent Decision Support System for Oil and Gas Drilling Operations

---

## Overview

This project is a **Streamlit-based Intelligent Decision Support System (IDSS)** built to analyze, classify, predict, and support decisions in oil and gas drilling operations. It integrates four real-world datasets covering drilling sensor logs, oil well reservoir properties, equipment maintenance records, and energy consumption data into a unified decision-support dashboard.

Rather than displaying raw data alone, the system applies a full IDSS pipeline:

> **Data Management → Model Management → Knowledge Base → Inference Engine → Decision Analysis → Explanation Facility**

The system is designed to answer operational questions such as:

> "What is the geological risk at the current drilling depth?"
> "Is equipment failure imminent, and should drilling be halted?"
> "Does this reservoir have viable oil potential?"
> "What is the optimal drilling decision given current risk conditions?"

---

## Project Motivation

Drilling operations in the oil and gas sector involve continuous decision-making under uncertainty. Formation changes, equipment degradation, reservoir uncertainty, and energy costs all interact simultaneously. Human operators must integrate signals from dozens of sensors and logs in real time.

This IDSS prototype demonstrates how structured decision-support methods — including machine learning, AHP weighting, goal programming, discriminant analysis, and classical decision theory — can be combined into one explainable dashboard to support better, faster, and more consistent operational decisions.

---

## Datasets

| Dataset | Source | Records | Purpose |
|---|---|---|---|
| Drilling Sensor Logs | FORCE 2020 Well Log Competition | 21 wells, ~400,000+ depth intervals | Formation classification, risk profiling |
| Oil Well Log Data | Synthetic geological dataset | 5,000 locations | Oil presence prediction, reservoir assessment |
| Equipment Failure Data | AI4I 2020 Predictive Maintenance Dataset | 10,000 records | Failure prediction, maintenance alerts |
| Energy and Fuel Consumption | Egypt national energy statistics | 60 records (2008–2022) | Energy trend analysis, efficiency scoring |

---

## System Architecture

The system is organized into six functional layers:

```
Raw Data (4 datasets)
        |
        v
Data Pipeline  [src/data_pipeline.py]
  Load, clean, merge, impute, encode
        |
        v
Model Layer    [src/models.py]
  Formation Classifier      (Random Forest, 95.8% accuracy)
  Oil Presence Predictor    (Gradient Boosting, 90.8% accuracy)
  Equipment Failure Model   (Random Forest, 98.7% accuracy)
        |
        v
Decision Analysis Layer    [src/decision_analysis.py]  [src/ahp.py]  [src/goal_programming.py]
  Discriminant Analysis     (LDA risk tiers, 86.6% accuracy)
  AHP Risk Weight Selection (Pairwise comparison + sensitivity analysis)
  Goal Programming          (Deviation minimization across 4 goals)
  Decision Under Certainty  (Utility maximization)
  Decision Under Risk       (Expected Monetary Value)
  Decision Under Uncertainty(Maximin, Maximax, Hurwicz, Minimax Regret, Laplace)
        |
        v
Knowledge Base + Inference Engine    [src/knowledge_base.py]
  14 domain rules across formation, equipment, oil potential, and energy
        |
        v
Streamlit Dashboard    [app.py]
  7 interactive pages with live simulators, charts, and explanation cards
```

---

## IDSS Methods Implemented

### Analytic Hierarchy Process (AHP)
Used for risk criterion weight selection. Decision-makers provide pairwise comparisons across four criteria — Geological Risk, Equipment Risk, Oil Potential, and Energy Efficiency — using the Saaty scale. The system computes priority weights, validates consistency via the Consistency Ratio (CR < 0.10), and runs sensitivity analysis to show how weights shift under different judgment scenarios.

### Goal Programming
Models competing operational goals simultaneously. Targets are set for drilling rate of penetration, equipment failure probability, energy efficiency, and oil potential score. The system quantifies deviation from each target, identifies the most critical gap, and issues a prioritized recommendation.

### Discriminant Analysis (LDA)
Applies Linear Discriminant Analysis to well log measurements (GR, RHOB, NPHI, RDEP, ROP, MUDWEIGHT, DTC, PEF) to classify each depth interval as LOW, MEDIUM, or HIGH geological risk. The discriminant coefficients reveal which measurements contribute most to risk separation.

### Decision Under Certainty
Assigns deterministic utility scores to five drilling alternatives (Continue Drilling, Adjust Mud Weight, Reduce ROP, Pull Out and Inspect, Sidetrack) based on current DSS readings. The alternative with the highest utility is selected.

### Decision Under Risk (EMV)
Estimates scenario probabilities (Stable Formation, Minor Hazard, Major Hazard, Equipment Failure) from live DSS readings, then computes the Expected Monetary Value for each alternative using a domain-defined payoff table. The highest EMV alternative is recommended.

### Decision Under Uncertainty
When probabilities are unknown, five classical uncertainty criteria are applied simultaneously: Maximin (pessimistic), Maximax (optimistic), Hurwicz (balanced), Minimax Regret (opportunity loss), and Laplace (equal probability). A consensus recommendation is determined from the most frequently selected alternative across criteria.

### Rule-Based Inference Engine
Fourteen domain rules covering formation hazards, equipment thresholds, oil potential signals, and energy anomalies are evaluated against live readings. Each fired rule produces a labeled alert with risk level and recommended action.

---

## Project Structure

```
Project/
│
├── app.py                        Main Streamlit dashboard (7 pages)
├── requirements.txt              Python dependencies
│
├── src/
│   ├── __init__.py
│   ├── data_pipeline.py          Data loading, cleaning, merging for all 4 datasets
│   ├── models.py                 ML models: formation, oil presence, equipment failure
│   ├── ahp.py                    AHP risk weight selection + sensitivity analysis
│   ├── goal_programming.py       Goal programming deviation analysis
│   ├── decision_analysis.py      LDA + decision under certainty/risk/uncertainty
│   └── knowledge_base.py         Domain rules and inference engine
│
└── data/
    ├── 15_9_14_well.csv          (and 20 other well log CSV files)
    ├── synthetic_oil_data.csv
    ├── ai4i2020.csv
    └── Energy___Fuel_Consumption_Data.csv
```

---

## Dashboard Pages

| Page | Content |
|---|---|
| Overview and KPIs | Dataset statistics, well log depth profiles, formation distribution |
| Formation Analysis | RF feature importance, GR vs RHOB cross-plot, LDA risk tier distribution |
| Equipment Health | Live failure risk gauge, inference alerts, tool wear analysis |
| Oil Potential | Reservoir predictor simulator, oil presence probability, feature importance |
| Energy and Fuel | Egypt consumption trends (2008–2022), year-over-year change, energy alerts |
| IDSS Decision Center | AHP weights, goal programming simulation, composite score, full inference summary |
| Decision Analysis | LDA discriminant analysis, AHP with sensitivity, certainty/risk/uncertainty tabs |

---

## Model Performance Summary

| Model | Method | Accuracy |
|---|---|---|
| Formation Classifier | Random Forest | 95.8% |
| Oil Presence Predictor | Gradient Boosting | 90.8% |
| Equipment Failure Predictor | Random Forest | 98.7% |
| Risk Tier Classifier | Linear Discriminant Analysis | 86.6% |

---

## Installation and Setup

**Step 1 — Clone the repository**
```bash
git clone https://github.com/your-username/Drilling-Intelligence-DSS.git
cd Drilling-Intelligence-DSS
```

**Step 2 — Place your data files in the `data/` folder**

All 21 well log CSV files plus the three supporting datasets should be placed in `data/`.

**Step 3 — Install dependencies**
```bash
pip install -r requirements.txt
```

**Step 4 — Run the dashboard**
```bash
streamlit run app.py
```

On Windows:
```powershell
py -3 -m pip install -r requirements.txt
py -3 -m streamlit run app.py
```

---

## Dependencies

```
streamlit>=1.35.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
plotly>=5.18.0
scipy>=1.11.0
openpyxl>=3.1.0
```

---

## Educational Scope

This project was developed for an Intelligent Decision Support Systems (IDSS) course. It demonstrates how the classical IDSS architecture — data management, model management, knowledge base, inference engine, and explanation facility — can be applied to a real-world engineering domain.

The system uses real well log data from the FORCE 2020 competition and a publicly available predictive maintenance dataset. Some decision payoff values and rule thresholds are calibrated for educational demonstration purposes and would require domain expert validation before deployment in a production environment.

---

## Key References

- FORCE 2020 Well Log and Lithofacies Dataset — Equinor and partners
- AI4I 2020 Predictive Maintenance Dataset — UCI Machine Learning Repository
- Saaty, T.L. (1980). The Analytic Hierarchy Process. McGraw-Hill.
- Hwang, C.L. and Masud, A.S.M. (1979). Multiple Objective Decision Making. Springer.
- Turban, E., Aronson, J.E. and Liang, T.P. (2005). Decision Support Systems and Intelligent Systems. Pearson.
