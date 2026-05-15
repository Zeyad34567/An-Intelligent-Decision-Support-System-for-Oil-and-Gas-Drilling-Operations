"""
Drilling Intelligence DSS — Streamlit Dashboard
Oil & Gas Intelligent Decision Support System
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from src.data_pipeline   import load_drilling_sensor, load_oil_well_log, load_equipment_failure, load_energy_data, get_well_summary
from src.models          import (train_formation_model, train_oil_model, train_equipment_model,
                                  train_discriminant_model, predict_oil_presence,
                                  predict_failure_risk, predict_risk_tier, FORMATION_FEATURES, OIL_FEATURES, EQUIP_FEATURES, RISK_FEATURES)
from src.ahp             import compute_ahp_weights, compute_ahp_score, build_pairwise_from_sliders, sensitivity_analysis, CRITERIA
from src.goal_programming import goal_programming_score
from src.knowledge_base  import run_all_inference
from src.decision_analysis import (train_discriminant_analysis, predict_risk_tier,
                                    classify_all_depths, decision_under_certainty,
                                    compute_certainty_scores, decision_under_risk,
                                    estimate_scenario_probabilities, decision_under_uncertainty,
                                    SCENARIOS, DRILLING_ALTERNATIVES)

# ─── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Drilling Intelligence DSS",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title  { font-size:2.2rem; font-weight:700; color:#f0a500; margin-bottom:0.2rem; }
    .sub-title   { font-size:1rem;   color:#aaa; margin-bottom:1.5rem; }
    .kpi-card    { background:#1e2430; border-radius:10px; padding:1rem 1.2rem;
                   border-left:4px solid #f0a500; margin-bottom:0.5rem; }
    .kpi-value   { font-size:1.8rem; font-weight:700; color:#f0a500; }
    .kpi-label   { font-size:0.8rem; color:#aaa; text-transform:uppercase; letter-spacing:1px; }
    .alert-high  { background:#3d1a1a; border-left:4px solid #e74c3c; border-radius:6px;
                   padding:0.7rem 1rem; margin:0.3rem 0; }
    .alert-med   { background:#3d2e1a; border-left:4px solid #f39c12; border-radius:6px;
                   padding:0.7rem 1rem; margin:0.3rem 0; }
    .alert-low   { background:#1a3d2b; border-left:4px solid #27ae60; border-radius:6px;
                   padding:0.7rem 1rem; margin:0.3rem 0; }
    .section-hdr { font-size:1.2rem; font-weight:600; color:#f0a500;
                   border-bottom:1px solid #333; padding-bottom:0.4rem; margin:1rem 0 0.8rem 0; }
    .explain-box { background:#1a2233; border:1px solid #2d4a6b; border-radius:8px;
                   padding:1rem; margin-top:0.5rem; font-size:0.9rem; color:#ccc; }
</style>
""", unsafe_allow_html=True)

# ─── Data + Model Loading (cached) ─────────────────────────────────────────────
@st.cache_data(show_spinner="Loading datasets...")
def load_all_data():
    return (
        load_drilling_sensor(),
        load_oil_well_log(),
        load_equipment_failure(),
        load_energy_data(),
    )

@st.cache_resource(show_spinner="Training models...")
def load_all_models(df_drill, df_oil, df_equip):
    m_form,  le_form,  r_form  = train_formation_model(df_drill)
    m_oil,   sc_oil,   r_oil   = train_oil_model(df_oil)
    m_equip, sc_equip, r_equip = train_equipment_model(df_equip)
    m_lda,   sc_lda, ft_lda, r_lda = train_discriminant_model(df_drill)
    return (m_form, le_form, r_form,
            m_oil,  sc_oil,  r_oil,
            m_equip,sc_equip,r_equip,
            m_lda,  sc_lda,  ft_lda, r_lda)

df_drill, df_oil, df_equip, df_energy = load_all_data()
(m_form, le_form, r_form,
 m_oil,  sc_oil,  r_oil,
 m_equip,sc_equip,r_equip,
 m_lda,  sc_lda,  ft_lda, r_lda) = load_all_models(df_drill, df_oil, df_equip)

# ─── Sidebar Navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛢️ Drilling DSS")
    st.markdown("---")
    page = st.radio("Navigation", [
        "📊 Overview & KPIs",
        "🪨 Formation Analysis",
        "🛠️ Equipment Health",
        "🛢️ Oil Potential",
        "⚡ Energy & Fuel",
        "🧠 IDSS Decision Center",
        "📐 Decision Analysis",
    ])
    st.markdown("---")
    st.markdown("**Model Accuracies**")
    st.markdown(f"Formation:  `{r_form['accuracy']:.1%}`")
    st.markdown(f"Oil Presence: `{r_oil['accuracy']:.1%}`")
    st.markdown(f"Equipment: `{r_equip['accuracy']:.1%}`")
    st.markdown(f"Risk Tiers (LDA): `{r_lda['accuracy']:.1%}`")
    st.markdown("---")
    st.caption("IDSS Course Project | Oil & Gas Drilling")

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 1: Overview & KPIs
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Overview & KPIs":
    st.markdown('<div class="main-title">🛢️ Drilling Intelligence DSS</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Intelligent Decision Support System for Oil & Gas Drilling Operations</div>', unsafe_allow_html=True)

    well_summary = get_well_summary(df_drill)
    failure_rate = df_equip["Machine_Failure"].mean()
    oil_rate     = df_oil["Oil_Presence"].mean()

    # KPI cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-value">{len(df_drill):,}</div>
            <div class="kpi-label">Depth Readings</div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-value">{df_drill['FORMATION'].nunique()}</div>
            <div class="kpi-label">Formations Identified</div></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-value">{failure_rate:.1%}</div>
            <div class="kpi-label">Equipment Failure Rate</div></div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-value">{oil_rate:.1%}</div>
            <div class="kpi-label">Oil Presence Rate</div></div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-hdr">Well Log Overview — Depth Profiles</div>', unsafe_allow_html=True)
    col_left, col_right = st.columns(2)
    with col_left:
        fig = px.line(df_drill.head(500), x="GR", y="DEPTH_MD",
                      title="Gamma Ray vs Depth",
                      labels={"GR": "Gamma Ray (API)", "DEPTH_MD": "Depth (m)"},
                      color_discrete_sequence=["#f0a500"])
        fig.update_yaxes(autorange="reversed")
        fig.update_layout(template="plotly_dark", height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        fig = px.line(df_drill.head(500), x="RHOB", y="DEPTH_MD",
                      title="Bulk Density vs Depth",
                      labels={"RHOB": "RHOB (g/cc)", "DEPTH_MD": "Depth (m)"},
                      color_discrete_sequence=["#3498db"])
        fig.update_yaxes(autorange="reversed")
        fig.update_layout(template="plotly_dark", height=400)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-hdr">Formation Distribution</div>', unsafe_allow_html=True)
    form_counts = df_drill["FORMATION"].value_counts().head(10).reset_index()
    form_counts.columns = ["Formation", "Count"]
    fig = px.bar(form_counts, x="Formation", y="Count",
                 color="Count", color_continuous_scale="Oranges",
                 title="Top 10 Formations by Depth Reading Count")
    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 2: Formation Analysis
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🪨 Formation Analysis":
    st.markdown('<div class="main-title">🪨 Formation Analysis</div>', unsafe_allow_html=True)
    st.markdown("Lithology classification using well-log measurements (GR, RHOB, NPHI, RDEP, DTC).")

    st.markdown('<div class="section-hdr">Feature Importance (Random Forest)</div>', unsafe_allow_html=True)
    fi = r_form["feature_importance"].reset_index()
    fi.columns = ["Feature", "Importance"]
    fig = px.bar(fi, x="Importance", y="Feature", orientation="h",
                 color="Importance", color_continuous_scale="Oranges",
                 title=f"Formation Classifier — Feature Importance (Accuracy: {r_form['accuracy']:.1%})")
    fig.update_layout(template="plotly_dark", yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-hdr">Cross-Plot: GR vs RHOB (coloured by Formation)</div>', unsafe_allow_html=True)
    sample = df_drill[df_drill["FORMATION"] != "Unknown"].sample(min(1000, len(df_drill)))
    fig = px.scatter(sample, x="GR", y="RHOB", color="FORMATION",
                     title="GR vs RHOB — Formation Discrimination",
                     labels={"GR": "Gamma Ray (API)", "RHOB": "Bulk Density (g/cc)"})
    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-hdr">Risk Tier Distribution (LDA Discriminant Analysis)</div>', unsafe_allow_html=True)
    features = [f for f in RISK_FEATURES if f in df_drill.columns]
    X_sample = df_drill[features].dropna().head(2000)
    X_s = sc_lda.transform(X_sample)
    risk_preds = m_lda.predict(X_s)
    risk_df = pd.DataFrame({"Risk_Tier": risk_preds, "Depth": df_drill.dropna(subset=features)["DEPTH_MD"].head(2000).values})

    col1, col2 = st.columns(2)
    with col1:
        rc = pd.Series(risk_preds).value_counts().reset_index()
        rc.columns = ["Tier", "Count"]
        colors = {"LOW": "#27ae60", "MEDIUM": "#f39c12", "HIGH": "#e74c3c"}
        rc["Color"] = rc["Tier"].map(colors)
        fig = px.pie(rc, names="Tier", values="Count",
                     color="Tier", color_discrete_map=colors,
                     title="Risk Tier Distribution (LDA)")
        fig.update_layout(template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.scatter(risk_df, x=range(len(risk_df)), y="Depth", color="Risk_Tier",
                         color_discrete_map=colors,
                         title="Risk Tier by Depth",
                         labels={"x": "Sample Index", "Depth": "Depth (m)"})
        fig.update_yaxes(autorange="reversed")
        fig.update_layout(template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-hdr">📖 Explanation: How LDA Risk Tiers Work</div>', unsafe_allow_html=True)
    st.markdown("""<div class="explain-box">
    <b>Linear Discriminant Analysis (LDA)</b> finds the linear combination of well-log features
    that best separates depth intervals into risk classes.<br><br>
    <b>Rule-based labels used for training:</b><br>
    • HIGH GR (>75 API) → shale indicator → elevated risk<br>
    • Low RHOB (<2.2 g/cc) → porous zone → medium risk<br>
    • High NPHI (>0.3) → fluid-bearing zone → medium risk<br>
    • Low RDEP (<2 Ω·m) → wet zone or brine → high risk<br>
    • High ROP (>80 m/hr) → soft/unstable formation → medium risk<br><br>
    Risk classes: <span style="color:#e74c3c">HIGH</span> (3+ risk flags) |
    <span style="color:#f39c12">MEDIUM</span> (1–2 flags) |
    <span style="color:#27ae60">LOW</span> (0 flags)
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 3: Equipment Health
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🛠️ Equipment Health":
    st.markdown('<div class="main-title">🛠️ Equipment Health & Predictive Maintenance</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-hdr">Live Equipment Risk Simulator</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        air_temp   = st.slider("Air Temp (K)",     295.0, 305.0, 298.0, 0.1)
        proc_temp  = st.slider("Process Temp (K)", 305.0, 315.0, 308.0, 0.1)
        tool_wear  = st.slider("Tool Wear (min)",  0, 250, 80)
    with col2:
        torque     = st.slider("Torque (Nm)",      20.0, 80.0, 40.0, 0.5)
        rot_speed  = st.slider("Rot. Speed (rpm)", 1000, 2000, 1500)
        mtype      = st.selectbox("Machine Type",  ["L", "M", "H"])
    with col3:
        type_code  = {"L": 0, "M": 1, "H": 2}[mtype]
        temp_diff  = proc_temp - air_temp
        power      = torque * rot_speed / 9550
        st.metric("Temp Differential", f"{temp_diff:.1f} K")
        st.metric("Power Proxy",        f"{power:.2f} kW")

    reading = {
        "Air_Temp_K": air_temp, "Process_Temp_K": proc_temp,
        "Rot_Speed_rpm": rot_speed, "Torque_Nm": torque,
        "Tool_Wear_min": tool_wear, "Temp_Diff": temp_diff,
        "Power": power, "Type_Code": type_code,
    }
    fail_prob = predict_failure_risk(m_equip, sc_equip, reading)

    col_gauge, col_alerts = st.columns([1, 1])
    with col_gauge:
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=fail_prob * 100,
            title={"text": "Failure Risk %", "font": {"color": "white"}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar":  {"color": "#e74c3c" if fail_prob > 0.5 else "#f39c12" if fail_prob > 0.2 else "#27ae60"},
                "steps": [
                    {"range": [0, 20],  "color": "#1a3d2b"},
                    {"range": [20, 50], "color": "#3d2e1a"},
                    {"range": [50, 100],"color": "#3d1a1a"},
                ],
                "threshold": {"line": {"color": "white", "width": 2}, "value": 50},
            },
            delta={"reference": 20},
            number={"suffix": "%", "font": {"color": "white"}},
        ))
        fig.update_layout(template="plotly_dark", height=300)
        st.plotly_chart(fig, use_container_width=True)

    with col_alerts:
        st.markdown("**Inference Engine — Equipment Rules:**")
        equip_alerts = run_all_inference(reading).get("equipment_rules", [])
        if equip_alerts:
            for a in equip_alerts:
                cls = "alert-high" if a["risk"] == "HIGH" else "alert-med" if a["risk"] == "MEDIUM" else "alert-low"
                st.markdown(f'<div class="{cls}"><b>[{a["risk"]}] {a["label"]}</b><br>{a["action"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="alert-low">✅ No equipment alerts triggered. All readings within normal range.</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-hdr">Feature Importance — Equipment Failure Model</div>', unsafe_allow_html=True)
    fi = r_equip["feature_importance"].reset_index()
    fi.columns = ["Feature", "Importance"]
    fig = px.bar(fi, x="Feature", y="Importance", color="Importance",
                 color_continuous_scale="Reds",
                 title=f"Equipment Failure Model — Feature Importance (Accuracy: {r_equip['accuracy']:.1%})")
    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-hdr">Tool Wear Distribution</div>', unsafe_allow_html=True)
    fig = px.histogram(df_equip, x="Tool_Wear_min", color="Machine_Failure",
                       barmode="overlay", nbins=50,
                       color_discrete_map={0: "#27ae60", 1: "#e74c3c"},
                       labels={"Tool_Wear_min": "Tool Wear (min)", "Machine_Failure": "Failure"},
                       title="Tool Wear Distribution — Failures vs Non-Failures")
    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 4: Oil Potential
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🛢️ Oil Potential":
    st.markdown('<div class="main-title">🛢️ Oil Potential Assessment</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-hdr">Reservoir Predictor Simulator</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        rock      = st.selectbox("Rock Type", ["Sandstone", "Limestone", "Shale"])
        porosity  = st.slider("Porosity (%)",       0.0, 30.0, 15.0, 0.5)
        perm      = st.slider("Permeability (mD)",  1.0, 1000.0, 200.0, 1.0)
        trap      = st.selectbox("Trap Type",       ["Anticline", "Dome", "Fault", "None"])
    with col2:
        seismic   = st.slider("Seismic Score",      0.0, 1.0, 0.6, 0.01)
        proximity = st.slider("Proximity to Oil Field (km)", 0.0, 50.0, 5.0, 0.5)
        depth     = st.slider("Estimated Reservoir Depth (m)", 500, 5000, 2000, 100)

    rock_map = {"Sandstone": 2, "Limestone": 1, "Shale": 0}
    trap_map = {"Anticline": 3, "Dome": 2, "Fault": 1, "None": 0}
    oil_input = {
        "Rock_Type_Code": rock_map[rock], "Porosity": porosity,
        "Permeability": perm, "Trap_Type_Code": trap_map[trap],
        "Seismic_Score": seismic, "Proximity_to_Oil_Field": proximity,
        "Estimated_Reservoir_Depth": depth,
    }
    oil_pred, oil_prob = predict_oil_presence(m_oil, sc_oil, oil_input)

    col_res, col_info = st.columns(2)
    with col_res:
        color = "#27ae60" if oil_prob > 0.6 else "#f39c12" if oil_prob > 0.35 else "#e74c3c"
        verdict = "OIL LIKELY PRESENT ✅" if oil_prob > 0.6 else "UNCERTAIN ⚠️" if oil_prob > 0.35 else "OIL UNLIKELY ❌"
        st.markdown(f"""<div class="kpi-card" style="border-left-color:{color}">
            <div class="kpi-value" style="color:{color}">{oil_prob:.1%}</div>
            <div class="kpi-label">Oil Presence Probability</div>
            <div style="color:{color}; font-weight:600; margin-top:0.5rem">{verdict}</div>
        </div>""", unsafe_allow_html=True)

    with col_info:
        oil_alerts = run_all_inference({**oil_input, "Oil_Presence": oil_pred}).get("oil_potential_rules", [])
        st.markdown("**Knowledge Base — Oil Potential Rules:**")
        if oil_alerts:
            for a in oil_alerts:
                cls = "alert-high" if a["risk"] == "HIGH" else "alert-med" if a["risk"] == "MEDIUM" else "alert-low"
                st.markdown(f'<div class="{cls}"><b>[{a["risk"]}] {a["label"]}</b><br>{a["action"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="alert-med">No specific oil potential rules triggered for this configuration.</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-hdr">Feature Importance — Oil Presence Model</div>', unsafe_allow_html=True)
    fi = r_oil["feature_importance"].reset_index()
    fi.columns = ["Feature", "Importance"]
    fig = px.bar(fi, x="Feature", y="Importance", color="Importance",
                 color_continuous_scale="YlOrBr",
                 title=f"Oil Presence Model — Feature Importance (Accuracy: {r_oil['accuracy']:.1%})")
    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        fig = px.scatter(df_oil.sample(500), x="Porosity", y="Permeability",
                         color=df_oil.sample(500)["Oil_Presence"].map({0:"No Oil", 1:"Oil"}),
                         color_discrete_map={"No Oil": "#e74c3c", "Oil": "#27ae60"},
                         title="Porosity vs Permeability — Oil Presence")
        fig.update_layout(template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        fig = px.histogram(df_oil, x="Seismic_Score", color=df_oil["Oil_Presence"].map({0:"No Oil", 1:"Oil"}),
                           barmode="overlay", nbins=40,
                           color_discrete_map={"No Oil": "#e74c3c", "Oil": "#27ae60"},
                           title="Seismic Score Distribution")
        fig.update_layout(template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 5: Energy & Fuel
# ══════════════════════════════════════════════════════════════════════════════
elif page == "⚡ Energy & Fuel":
    st.markdown('<div class="main-title">⚡ Energy & Fuel Consumption</div>', unsafe_allow_html=True)

    types = df_energy["Energy_Type"].unique()
    selected = st.multiselect("Select Energy Types", list(types), default=list(types))
    df_filt = df_energy[df_energy["Energy_Type"].isin(selected)]

    fig = px.line(df_filt, x="Year", y="Consumption", color="Energy_Type",
                  title="Egypt Energy Consumption by Type (2008–2022)",
                  labels={"Consumption": "Consumption (ktoe)", "Year": "Year"},
                  markers=True, color_discrete_sequence=px.colors.qualitative.Set2)
    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-hdr">Year-over-Year Change</div>', unsafe_allow_html=True)
    df_pivot = df_energy.pivot(index="Year", columns="Energy_Type", values="Consumption").fillna(0)
    df_yoy   = df_pivot.pct_change().dropna() * 100
    df_yoy_m = df_yoy.reset_index().melt(id_vars="Year", var_name="Energy_Type", value_name="YoY_Change")
    fig2 = px.bar(df_yoy_m, x="Year", y="YoY_Change", color="Energy_Type", barmode="group",
                  title="YoY % Change in Energy Consumption",
                  labels={"YoY_Change": "% Change"},
                  color_discrete_sequence=px.colors.qualitative.Set2)
    fig2.add_hline(y=15, line_dash="dash", line_color="red", annotation_text="Alert Threshold +15%")
    fig2.update_layout(template="plotly_dark")
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="section-hdr">Energy Rules — Knowledge Base</div>', unsafe_allow_html=True)
    oil_data = df_energy[df_energy["Energy_Type"] == "Oil"].sort_values("Year")
    if len(oil_data) >= 2:
        latest_oil = oil_data.iloc[-1]["Consumption"]
        prev_oil   = oil_data.iloc[-2]["Consumption"]
        yoy_change = (latest_oil - prev_oil) / (prev_oil + 1e-9)
        gas_latest = df_energy[df_energy["Energy_Type"] == "Gas"]["Consumption"].iloc[-1]
        energy_reading = {"Oil_YoY_Change": yoy_change, "Gas_Consumption": gas_latest}
        energy_alerts  = run_all_inference(energy_reading).get("energy_rules", [])
        for a in energy_alerts:
            cls = "alert-high" if a["risk"] == "HIGH" else "alert-med" if a["risk"] == "MEDIUM" else "alert-low"
            st.markdown(f'<div class="{cls}"><b>[{a["risk"]}] {a["label"]}</b><br>{a["action"]}</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 6: IDSS Decision Center
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🧠 IDSS Decision Center":
    st.markdown('<div class="main-title">🧠 IDSS Decision Center</div>', unsafe_allow_html=True)
    st.markdown("Composite decision analysis combining AHP weighting, Goal Programming, and all inference modules.")

    st.markdown('<div class="section-hdr">Step 1 — AHP Criteria Weights</div>', unsafe_allow_html=True)
    ahp_result = compute_ahp_weights()
    ahp_df = pd.DataFrame({"Criterion": list(ahp_result["weights"].keys()),
                            "Weight":    list(ahp_result["weights"].values())})
    col1, col2 = st.columns([1, 1])
    with col1:
        fig = px.bar(ahp_df, x="Criterion", y="Weight", color="Weight",
                     color_continuous_scale="Oranges",
                     title=f"AHP Priority Weights (CR={ahp_result['CR']:.4f} — {'✅ Consistent' if ahp_result['consistent'] else '❌ Inconsistent'})")
        fig.update_layout(template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.pie(ahp_df, names="Criterion", values="Weight",
                     color_discrete_sequence=px.colors.sequential.Oranges_r,
                     title="Weight Distribution")
        fig.update_layout(template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("""<div class="explain-box">
    <b>AHP Pairwise Comparison Results:</b><br>
    Oil Potential carries the highest weight because identifying productive reservoirs is the primary objective.<br>
    Geological Risk is second — formation instability directly impacts safety and cost.<br>
    Equipment Risk is third — maintenance failures cause costly downtime.<br>
    Energy Efficiency is fourth — important but secondary to core drilling objectives.<br>
    <b>Consistency Ratio (CR) < 0.10</b> confirms the judgment matrix is logically consistent.
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-hdr">Step 2 — Goal Programming Simulation</div>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1: rop_s = st.slider("ROP Score (0–1)",     0.0, 1.0, 0.6, 0.05)
    with col2: fail_s = st.slider("Failure Risk (0–1)", 0.0, 1.0, 0.3, 0.05)
    with col3: energy_s = st.slider("Energy Score (0–1)", 0.0, 1.0, 0.55, 0.05)
    with col4: oil_s = st.slider("Oil Potential (0–1)",   0.0, 1.0, 0.5, 0.05)

    gp = goal_programming_score(rop_s, fail_s, energy_s, oil_s)

    c1, c2, c3 = st.columns(3)
    c1.metric("Overall Performance", f"{gp['performance_score']:.1%}")
    c2.metric("Total Deviation",     f"{gp['total_deviation']:.4f}")
    c3.metric("Worst Goal",          gp['worst_goal'])

    dev_df = pd.DataFrame({"Goal": list(gp["deviations"].keys()),
                            "Deviation": list(gp["deviations"].values())})
    fig = px.bar(dev_df, x="Goal", y="Deviation", color="Deviation",
                 color_continuous_scale="RdYlGn_r",
                 title="Goal Programming — Deviation from Targets")
    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"""<div class="explain-box">
    <b>Goal Programming Recommendation:</b><br>
    {gp['recommendation']}
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-hdr">Step 3 — Composite AHP Score</div>', unsafe_allow_html=True)
    geo_risk_norm   = 1 - rop_s          # lower ROP → higher geological stress
    equip_risk_norm = 1 - fail_s         # lower failure risk is better
    oil_norm        = oil_s
    energy_norm     = energy_s

    ahp_score, _ = compute_ahp_score(geo_risk_norm, equip_risk_norm, oil_norm, energy_norm)
    color = "#27ae60" if ahp_score > 0.6 else "#f39c12" if ahp_score > 0.4 else "#e74c3c"
    verdict_map = {True: "PROCEED WITH DRILLING ✅", False: "REVIEW BEFORE PROCEEDING ⚠️"}
    st.markdown(f"""<div class="kpi-card" style="border-left-color:{color}">
        <div class="kpi-value" style="color:{color}">{ahp_score:.3f}</div>
        <div class="kpi-label">Composite AHP Decision Score (0–1)</div>
        <div style="color:{color}; font-weight:600; margin-top:0.5rem">
            {verdict_map[ahp_score > 0.5]}
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-hdr">Step 4 — Full Inference Summary</div>', unsafe_allow_html=True)
    full_readings = {
        "GR": 80, "RHOB": 2.2, "NPHI": 0.35, "RDEP": 1.5, "ROP": rop_s * 120,
        "Tool_Wear_min": fail_s * 250, "Torque_Nm": 40 + fail_s * 30,
        "Rot_Speed_rpm": 1800 - fail_s * 500, "Temp_Diff": 8 + fail_s * 8,
        "Power": 5 + fail_s * 5, "Porosity": oil_s * 25,
        "Permeability": oil_s * 800, "Oil_Presence": int(oil_s > 0.5),
        "Seismic_Score": oil_s, "Proximity_to_Oil_Field": max(0.5, (1 - oil_s) * 10),
        "Estimated_Reservoir_Depth": 1000 + (1 - oil_s) * 4000,
        "Oil_YoY_Change": 0.1 + (1 - energy_s) * 0.2,
        "Gas_Consumption": energy_s * 2000,
    }
    all_alerts = run_all_inference(full_readings)
    for cat, alerts in all_alerts.items():
        if alerts:
            st.markdown(f"**{cat.replace('_', ' ').title()}**")
            for a in alerts:
                cls = "alert-high" if a["risk"] == "HIGH" else "alert-med" if a["risk"] == "MEDIUM" else "alert-low"
                st.markdown(f'<div class="{cls}"><b>[{a["risk"]}] {a["label"]}</b> — {a["action"]}</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 7: Decision Analysis
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📐 Decision Analysis":
    st.markdown('<div class="main-title">📐 Decision Analysis Methods</div>', unsafe_allow_html=True)
    st.markdown("IDSS decision support using Discriminant Analysis, AHP Risk Weighting, and Decision Theory under Certainty / Risk / Uncertainty.")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Discriminant Analysis",
        "⚖️ AHP Risk Weights",
        "✅ Under Certainty",
        "🎲 Under Risk",
        "❓ Under Uncertainty",
    ])

    # ── Cache DA model ──────────────────────────────────────────────────────
    @st.cache_resource(show_spinner="Training Discriminant Analysis model...")
    def get_da_model(_df):
        return train_discriminant_analysis(_df)

    da_lda, da_scaler, da_feats, da_metrics = get_da_model(df_drill)

    # ────────────────────────────────────────────────────────────────────────
    with tab1:
        st.markdown('<div class="section-hdr">Linear Discriminant Analysis — Well Risk Classification</div>', unsafe_allow_html=True)
        st.markdown("""<div class="explain-box">
        <b>What is LDA?</b> Linear Discriminant Analysis finds linear combinations of well-log features
        that best separate depth intervals into risk classes (LOW / MEDIUM / HIGH).<br><br>
        <b>How risk labels are generated:</b> A domain rule system scores each depth interval
        (GR, RHOB, NPHI, RDEP, ROP, MUDWEIGHT) then LDA learns the decision boundaries from those scores.
        </div>""", unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        col1.metric("LDA Accuracy",   f"{da_metrics['accuracy']:.1%}")
        col2.metric("Features Used",  len(da_feats))
        col3.metric("Risk Classes",   len(da_metrics["classes"]))

        # Confusion matrix
        st.markdown('<div class="section-hdr">Confusion Matrix</div>', unsafe_allow_html=True)
        cm = da_metrics["confusion_matrix"]
        classes = ["LOW", "MEDIUM", "HIGH"]
        fig = go.Figure(go.Heatmap(
            z=cm, x=classes, y=classes,
            colorscale="RdYlGn_r",
            text=cm, texttemplate="%{text}",
            showscale=True,
        ))
        fig.update_layout(
            template="plotly_dark", height=350,
            xaxis_title="Predicted", yaxis_title="Actual",
            title="LDA Confusion Matrix — Risk Tier Classification"
        )
        st.plotly_chart(fig, use_container_width=True)

        # LDA coefficients heatmap
        st.markdown('<div class="section-hdr">LDA Discriminant Coefficients</div>', unsafe_allow_html=True)
        coef = da_metrics["coefficients"]
        fig2 = go.Figure(go.Heatmap(
            z=coef.values, x=coef.columns.tolist(), y=coef.index.tolist(),
            colorscale="RdBu", zmid=0,
            text=coef.values.round(3), texttemplate="%{text}",
        ))
        fig2.update_layout(template="plotly_dark", height=300,
                           title="Discriminant Function Coefficients (Higher absolute value = more discriminating)")
        st.plotly_chart(fig2, use_container_width=True)

        # Apply to full dataset
        st.markdown('<div class="section-hdr">Risk Tier Distribution Across All Wells</div>', unsafe_allow_html=True)
        risk_series = classify_all_depths(da_lda, da_scaler, da_feats, df_drill)
        df_risk = df_drill.copy()
        df_risk["Risk_Tier"] = risk_series.values

        col_a, col_b = st.columns(2)
        with col_a:
            rc = risk_series.value_counts().reset_index()
            rc.columns = ["Tier", "Count"]
            colors = {"LOW": "#27ae60", "MEDIUM": "#f39c12", "HIGH": "#e74c3c"}
            fig = px.pie(rc, names="Tier", values="Count",
                         color="Tier", color_discrete_map=colors,
                         title="Risk Tier Distribution (All Depth Intervals)")
            fig.update_layout(template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            sample_risk = df_risk.sample(min(1500, len(df_risk)))
            fig = px.scatter(sample_risk, x="GR", y="DEPTH_MD",
                             color="Risk_Tier", color_discrete_map=colors,
                             title="GR vs Depth — Coloured by Risk Tier",
                             labels={"GR": "Gamma Ray (API)", "DEPTH_MD": "Depth (m)"})
            fig.update_yaxes(autorange="reversed")
            fig.update_layout(template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

        # Live predictor
        st.markdown('<div class="section-hdr">Single Depth Reading Predictor</div>', unsafe_allow_html=True)
        cols = st.columns(len(da_feats))
        reading = {}
        defaults = {"GR":75,"RHOB":2.3,"NPHI":0.25,"RDEP":3.0,"ROP":60,"MUDWEIGHT":1.3,"DTC":80,"PEF":3.0}
        for i, feat in enumerate(da_feats):
            with cols[i]:
                reading[feat] = st.number_input(feat, value=float(defaults.get(feat, 1.0)), step=0.1, key=f"da_{feat}")

        tier, proba = predict_risk_tier(da_lda, da_scaler, da_feats, reading)
        color_map = {"LOW": "#27ae60", "MEDIUM": "#f39c12", "HIGH": "#e74c3c"}
        c = color_map.get(tier, "#aaa")
        st.markdown(f"""<div class="kpi-card" style="border-left-color:{c}">
            <div class="kpi-value" style="color:{c}">{tier} RISK</div>
            <div class="kpi-label">LDA Predicted Risk Tier</div>
            <div style="margin-top:0.5rem; color:#ccc">
                {'  |  '.join([f'{k}: {v:.1%}' for k,v in proba.items()])}
            </div></div>""", unsafe_allow_html=True)

    # ────────────────────────────────────────────────────────────────────────
    with tab2:
        st.markdown('<div class="section-hdr">AHP — Risk Criterion Weight Selection</div>', unsafe_allow_html=True)
        st.markdown("""<div class="explain-box">
        <b>AHP (Analytic Hierarchy Process)</b> uses pairwise comparisons between criteria to derive
        priority weights. Adjust the sliders below to set how important each criterion is relative to another.
        The <b>Consistency Ratio (CR)</b> must be below 0.10 for the judgments to be valid.
        </div>""", unsafe_allow_html=True)

        st.markdown("**Pairwise Comparisons (Saaty Scale 1–9)**")
        st.caption("1 = equal importance, 3 = moderate, 5 = strong, 7 = very strong, 9 = extreme")

        c1, c2, c3 = st.columns(3)
        with c1:
            g_e = st.slider("Geological vs Equipment",  1, 9, 2, key="ahp_ge")
            g_o = st.slider("Geological vs Oil",        1, 9, 3, key="ahp_go")
            g_en= st.slider("Geological vs Energy",     1, 9, 3, key="ahp_gen")
        with c2:
            e_o = st.slider("Equipment vs Oil",         1, 9, 4, key="ahp_eo")
            e_en= st.slider("Equipment vs Energy",      1, 9, 2, key="ahp_een")
        with c3:
            o_en= st.slider("Oil vs Energy",            1, 9, 5, key="ahp_oen")

        custom_matrix = build_pairwise_from_sliders(g_e, 1/g_o, g_en, 1/e_o, e_en, o_en)
        ahp_r = compute_ahp_weights(custom_matrix)

        cr_color = "#27ae60" if ahp_r["consistent"] else "#e74c3c"
        st.markdown(f"""<div class="kpi-card" style="border-left-color:{cr_color}">
            <div class="kpi-value" style="color:{cr_color}">CR = {ahp_r['CR']:.4f}</div>
            <div class="kpi-label">{'✅ Consistent — judgments are valid (CR < 0.10)' if ahp_r['consistent'] else '❌ Inconsistent — please revise comparisons (CR ≥ 0.10)'}</div>
        </div>""", unsafe_allow_html=True)

        col_bar, col_pie = st.columns(2)
        w_df = pd.DataFrame({"Criterion": list(ahp_r["weights"].keys()),
                              "Weight":    list(ahp_r["weights"].values())})
        with col_bar:
            fig = px.bar(w_df, x="Criterion", y="Weight", color="Weight",
                         color_continuous_scale="Oranges",
                         title=f"AHP Priority Weights (λ_max={ahp_r['lambda_max']})")
            fig.update_layout(template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
        with col_pie:
            fig = px.pie(w_df, names="Criterion", values="Weight",
                         color_discrete_sequence=px.colors.sequential.Oranges_r,
                         title="Weight Distribution")
            fig.update_layout(template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="section-hdr">Sensitivity Analysis — How Weights Change</div>', unsafe_allow_html=True)
        sa_df = sensitivity_analysis(custom_matrix)
        fig = px.bar(sa_df.melt(id_vars="Scenario", value_vars=CRITERIA, var_name="Criterion", value_name="Weight"),
                     x="Criterion", y="Weight", color="Scenario", barmode="group",
                     title="AHP Sensitivity: How Weights Shift Across Scenarios")
        fig.update_layout(template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(sa_df.style.format({c: "{:.4f}" for c in CRITERIA + ["CR"]}), use_container_width=True)

    # ────────────────────────────────────────────────────────────────────────
    with tab3:
        st.markdown('<div class="section-hdr">Decision Under Certainty</div>', unsafe_allow_html=True)
        st.markdown("""<div class="explain-box">
        <b>Decision Under Certainty:</b> All outcomes are fully known. Each drilling alternative
        is assigned a deterministic utility score based on current DSS readings.
        The alternative with the highest utility is the optimal choice.
        </div>""", unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col1: risk_c  = st.selectbox("Current Risk Tier",  ["LOW","MEDIUM","HIGH"], index=1, key="cert_risk")
        with col2: fail_c  = st.slider("Failure Probability",   0.0, 1.0, 0.3, 0.05, key="cert_fail")
        with col3: oil_c   = st.slider("Oil Potential (0–1)",   0.0, 1.0, 0.5, 0.05, key="cert_oil")
        rop_c = st.slider("ROP Score (0–1)", 0.0, 1.0, 0.6, 0.05, key="cert_rop")

        cert_scores = compute_certainty_scores(risk_c, fail_c, oil_c, rop_c)
        cert_result = decision_under_certainty(cert_scores)

        best_alt = cert_result["best_alternative"]
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-value" style="color:#27ae60">{best_alt}</div>
            <div class="kpi-label">Optimal Decision Under Certainty  (score: {cert_result['best_score']:.3f})</div>
        </div>""", unsafe_allow_html=True)

        fig = px.bar(cert_result["ranked_table"], x="Alternative", y="Utility_Score",
                     color="Utility_Score", color_continuous_scale="RdYlGn",
                     text="Recommendation",
                     title="Utility Scores — All Drilling Alternatives")
        fig.update_traces(textposition="outside")
        fig.update_layout(template="plotly_dark", height=380)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(f'<div class="explain-box">{cert_result["explanation"]}</div>', unsafe_allow_html=True)

    # ────────────────────────────────────────────────────────────────────────
    with tab4:
        st.markdown('<div class="section-hdr">Decision Under Risk (EMV)</div>', unsafe_allow_html=True)
        st.markdown("""<div class="explain-box">
        <b>Decision Under Risk:</b> Scenario probabilities are known (or estimated from DSS data).
        The <b>Expected Monetary Value (EMV)</b> = Σ (probability × payoff) for each alternative.
        The alternative with the highest EMV is chosen.
        </div>""", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1: risk_r  = st.selectbox("Current Risk Tier",  ["LOW","MEDIUM","HIGH"], index=1, key="risk_tier")
        with col2: fail_r  = st.slider("Failure Probability",   0.0, 1.0, 0.3, 0.05, key="risk_fail")

        auto_probs = estimate_scenario_probabilities(risk_r, fail_r)
        st.markdown("**Scenario Probabilities** (auto-estimated from DSS — adjust if needed):")
        p_cols = st.columns(4)
        probs_input = []
        for i, (sc, p) in enumerate(zip(SCENARIOS, auto_probs)):
            with p_cols[i]:
                probs_input.append(st.slider(sc, 0.0, 1.0, float(p), 0.01, key=f"prob_{i}"))

        # Normalise
        total_p = sum(probs_input)
        norm_probs = [round(p/total_p, 4) for p in probs_input] if total_p > 0 else [0.25]*4
        st.caption(f"Normalised: {dict(zip(SCENARIOS, norm_probs))}")

        risk_result = decision_under_risk(norm_probs)

        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-value" style="color:#27ae60">{risk_result['best_alternative']}</div>
            <div class="kpi-label">Best Alternative by EMV  ({risk_result['best_emv']:.4f})</div>
        </div>""", unsafe_allow_html=True)

        emv_df = pd.DataFrame({"Alternative": list(risk_result["emv_scores"].keys()),
                                "EMV":         list(risk_result["emv_scores"].values())})
        fig = px.bar(emv_df, x="Alternative", y="EMV",
                     color="EMV", color_continuous_scale="RdYlGn",
                     title="Expected Monetary Value (EMV) per Alternative")
        fig.add_hline(y=0, line_dash="dash", line_color="white", annotation_text="Break-even")
        fig.update_layout(template="plotly_dark", height=380)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="section-hdr">Full Payoff Table</div>', unsafe_allow_html=True)
        payoff_display = risk_result["payoff_table"].copy()
        st.dataframe(payoff_display.style
                     .background_gradient(cmap="RdYlGn", subset=SCENARIOS)
                     .format("{:.3f}"), use_container_width=True)

        st.markdown(f'<div class="explain-box">{risk_result["explanation"]}</div>', unsafe_allow_html=True)

    # ────────────────────────────────────────────────────────────────────────
    with tab5:
        st.markdown('<div class="section-hdr">Decision Under Uncertainty</div>', unsafe_allow_html=True)
        st.markdown("""<div class="explain-box">
        <b>Decision Under Uncertainty:</b> Scenario probabilities are completely unknown.
        Five classical criteria are applied, each reflecting a different attitude toward risk:<br><br>
        • <b>Maximin (Wald):</b> Pessimist — protect against the worst case<br>
        • <b>Maximax:</b> Optimist — go for the best possible outcome<br>
        • <b>Hurwicz:</b> Balanced — α controls optimism vs pessimism<br>
        • <b>Minimax Regret:</b> Minimise opportunity loss vs best possible choice<br>
        • <b>Laplace:</b> Equal probability assumed — highest average payoff wins
        </div>""", unsafe_allow_html=True)

        alpha = st.slider("Hurwicz α (0 = fully pessimistic, 1 = fully optimistic)", 0.0, 1.0, 0.5, 0.05)
        unc_result = decision_under_uncertainty(alpha)

        st.markdown("**Results by Criterion:**")
        winner_cols = st.columns(5)
        criterion_colors = {"Maximin":"#3498db","Maximax":"#e67e22","Hurwicz":"#9b59b6",
                            "Minimax_Regret":"#e74c3c","Laplace":"#27ae60"}
        for col, (criterion, winner) in zip(winner_cols, unc_result["criteria_winners"].items()):
            with col:
                c = criterion_colors.get(criterion, "#aaa")
                st.markdown(f"""<div class="kpi-card" style="border-left-color:{c}; padding:0.6rem 0.8rem">
                    <div style="color:{c}; font-weight:700; font-size:0.85rem">{criterion}</div>
                    <div style="color:#fff; font-size:0.9rem; margin-top:0.3rem">{winner}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown(f"""<div class="kpi-card" style="border-left-color:#f0a500; margin-top:0.8rem">
            <div class="kpi-value">🏆 {unc_result['consensus']}</div>
            <div class="kpi-label">Consensus Recommendation (most criteria agree)</div>
        </div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-hdr">Decision Summary Table</div>', unsafe_allow_html=True)
        summary = unc_result["summary_table"]
        st.dataframe(summary.style
                     .background_gradient(subset=["Maximin"], cmap="Blues")
                     .background_gradient(subset=["Maximax"], cmap="Oranges")
                     .background_gradient(subset=["Hurwicz"],  cmap="Purples")
                     .background_gradient(subset=["Laplace_Avg"], cmap="Greens")
                     .format({c: "{:.3f}" for c in ["Maximin","Maximax","Hurwicz","Max_Regret","Laplace_Avg"]}),
                     use_container_width=True)

        st.markdown('<div class="section-hdr">Regret Matrix (Opportunity Loss)</div>', unsafe_allow_html=True)
        regret = unc_result["regret_matrix"]
        fig = go.Figure(go.Heatmap(
            z=regret.values, x=regret.columns.tolist(), y=regret.index.tolist(),
            colorscale="Reds",
            text=regret.values.round(3), texttemplate="%{text}",
        ))
        fig.update_layout(template="plotly_dark", height=350,
                          title="Regret Matrix — Opportunity Loss per Alternative × Scenario")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="section-hdr">Method Explanations</div>', unsafe_allow_html=True)
        for criterion, explanation in unc_result["explanations"].items():
            c = criterion_colors.get(criterion, "#aaa")
            st.markdown(f'<div class="explain-box" style="border-color:{c}"><b style="color:{c}">{criterion}:</b> {explanation}</div>', unsafe_allow_html=True)