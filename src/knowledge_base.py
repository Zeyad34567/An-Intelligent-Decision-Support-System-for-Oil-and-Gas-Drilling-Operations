"""
Module 3c: Knowledge Base
Structured domain rules for oil & gas drilling risk assessment.
"""

KNOWLEDGE_BASE = {
    "formation_rules": [
        {
            "id": "F001",
            "condition": lambda r: r.get("GR", 0) > 75 and r.get("RHOB", 3) < 2.3,
            "label":     "Possible Shale Zone",
            "risk":      "MEDIUM",
            "action":    "Monitor mud weight; shale swelling may cause wellbore instability.",
        },
        {
            "id": "F002",
            "condition": lambda r: r.get("NPHI", 0) > 0.3 and r.get("RDEP", 10) < 2.0,
            "label":     "High Porosity / Possible Water Zone",
            "risk":      "HIGH",
            "action":    "Evaluate fluid type with resistivity ratio. Consider water influx risk.",
        },
        {
            "id": "F003",
            "condition": lambda r: r.get("RHOB", 3) > 2.6 and r.get("GR", 0) < 30,
            "label":     "Dense / Tight Carbonate",
            "risk":      "LOW",
            "action":    "Tight carbonate zone. May require stimulation for production.",
        },
        {
            "id": "F004",
            "condition": lambda r: r.get("ROP", 0) > 100,
            "label":     "Rapid Penetration — Soft Formation",
            "risk":      "MEDIUM",
            "action":    "Reduce WOB to prevent bit bounce and wellbore deviation.",
        },
        {
            "id": "F005",
            "condition": lambda r: r.get("RDEP", 10) > 10 and r.get("NPHI", 1) < 0.15,
            "label":     "Tight / Low-Porosity Reservoir",
            "risk":      "LOW",
            "action":    "Low porosity reservoir. Evaluate for unconventional potential.",
        },
    ],
    "equipment_rules": [
        {
            "id": "E001",
            "condition": lambda r: r.get("Tool_Wear_min", 0) > 200,
            "label":     "Critical Tool Wear",
            "risk":      "HIGH",
            "action":    "⚠️ Pull out of hole for bit replacement immediately.",
        },
        {
            "id": "E002",
            "condition": lambda r: r.get("Torque_Nm", 0) > 60 and r.get("Rot_Speed_rpm", 0) < 1200,
            "label":     "High Torque / Low Speed",
            "risk":      "HIGH",
            "action":    "Possible string differential sticking. Jar or circulate to free.",
        },
        {
            "id": "E003",
            "condition": lambda r: r.get("Temp_Diff", 0) > 12,
            "label":     "High Temperature Differential",
            "risk":      "MEDIUM",
            "action":    "Check cooling system and mud circulation rate.",
        },
        {
            "id": "E004",
            "condition": lambda r: r.get("Power", 0) > 8,
            "label":     "High Power Draw",
            "risk":      "MEDIUM",
            "action":    "Motor running above design power. Reduce rotary speed.",
        },
    ],
    "oil_potential_rules": [
        {
            "id": "O001",
            "condition": lambda r: r.get("Porosity", 0) > 15 and r.get("Permeability", 0) > 100 and r.get("Oil_Presence", 0) == 1,
            "label":     "High-Quality Reservoir",
            "risk":      "LOW",
            "action":    "✅ Excellent reservoir quality. Recommend production test.",
        },
        {
            "id": "O002",
            "condition": lambda r: r.get("Seismic_Score", 0) > 0.7 and r.get("Proximity_to_Oil_Field", 10) < 3,
            "label":     "High Seismic Score near Existing Field",
            "risk":      "LOW",
            "action":    "Strong geological analog. Prioritise for appraisal drilling.",
        },
        {
            "id": "O003",
            "condition": lambda r: r.get("Estimated_Reservoir_Depth", 0) > 4000,
            "label":     "Deep Reservoir — High Drilling Cost",
            "risk":      "MEDIUM",
            "action":    "Deep target. Conduct economic viability analysis before commitment.",
        },
    ],
    "energy_rules": [
        {
            "id": "EN001",
            "condition": lambda r: r.get("Oil_YoY_Change", 0) > 0.15,
            "label":     "Oil Consumption Spike",
            "risk":      "MEDIUM",
            "action":    "Oil consumption increased >15 % YoY. Audit fuel usage on rigs.",
        },
        {
            "id": "EN002",
            "condition": lambda r: r.get("Gas_Consumption", 0) < 500,
            "label":     "Low Gas Utilisation",
            "risk":      "LOW",
            "action":    "Consider increasing gas usage to reduce oil dependency.",
        },
    ],
}


def run_inference(readings: dict, rule_category: str) -> list:
    """
    Fire all rules in a category against a readings dict.
    Returns list of triggered rule dicts.
    """
    rules   = KNOWLEDGE_BASE.get(rule_category, [])
    alerts  = []
    for rule in rules:
        try:
            if rule["condition"](readings):
                alerts.append({
                    "id":     rule["id"],
                    "label":  rule["label"],
                    "risk":   rule["risk"],
                    "action": rule["action"],
                })
        except Exception:
            pass
    return alerts


def run_all_inference(readings: dict) -> dict:
    """Run inference across all rule categories."""
    return {
        cat: run_inference(readings, cat)
        for cat in KNOWLEDGE_BASE.keys()
    }


if __name__ == "__main__":
    sample = {
        "GR": 80, "RHOB": 2.2, "NPHI": 0.35, "RDEP": 1.5, "ROP": 110,
        "Tool_Wear_min": 220, "Torque_Nm": 65, "Rot_Speed_rpm": 1100,
        "Temp_Diff": 14, "Power": 9,
        "Porosity": 18, "Permeability": 200, "Oil_Presence": 1,
        "Seismic_Score": 0.8, "Proximity_to_Oil_Field": 2,
        "Estimated_Reservoir_Depth": 4500,
        "Oil_YoY_Change": 0.2, "Gas_Consumption": 400,
    }
    results = run_all_inference(sample)
    for cat, alerts in results.items():
        print(f"\n{cat.upper()}:")
        for a in alerts:
            print(f"  [{a['risk']}] {a['label']}: {a['action']}")