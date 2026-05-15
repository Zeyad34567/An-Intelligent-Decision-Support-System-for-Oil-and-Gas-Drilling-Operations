"""
Module 1: Data Pipeline
Loads, cleans, and prepares all 4 datasets for the Drilling DSS.
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"

def load_drilling_sensor():
    """
    Loads ALL well log CSV files from the data/ folder and merges them.
    Automatically skips non-well-log files (ai4i, synthetic_oil, energy).
    """
    EXCLUDE = {"ai4i2020.csv", "synthetic_oil_data.csv", "Energy___Fuel_Consumption_Data.csv"}
    well_files = [f for f in DATA_DIR.glob("*.csv") if f.name not in EXCLUDE]

    if not well_files:
        raise FileNotFoundError(f"No well CSV files found in {DATA_DIR}")

    frames = []
    loaded = []
    for f in sorted(well_files):
        try:
            tmp = pd.read_csv(f)
            # Only accept files that look like well logs
            if "DEPTH_MD" in tmp.columns and "GR" in tmp.columns:
                if "WELL" not in tmp.columns:
                    tmp["WELL"] = f.stem  # use filename as well name
                frames.append(tmp)
                loaded.append(f.name)
        except Exception:
            pass

    if not frames:
        raise ValueError("No valid well log files could be loaded.")

    print(f"Loaded {len(frames)} well files: {loaded}")
    df = pd.concat(frames, ignore_index=True)

    # Drop columns with >80% missing across merged dataset
    missing_ratio = df.isnull().mean()
    drop_cols = missing_ratio[missing_ratio > 0.80].index.tolist()
    df = df.drop(columns=drop_cols)

    # Fill numeric NaNs with per-column median
    num_cols = df.select_dtypes(include="number").columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    df["FORMATION"] = df["FORMATION"].fillna("Unknown")
    df["GROUP"]     = df["GROUP"].fillna("Unknown")

    return df

def load_oil_well_log():
    path = DATA_DIR / "synthetic_oil_data.csv"
    df = pd.read_csv(path)
    df["Trap_Type"] = df["Trap_Type"].fillna("None")
    rock_map  = {"Sandstone": 2, "Limestone": 1, "Shale": 0}
    trap_map  = {"Anticline": 3, "Dome": 2, "Fault": 1, "None": 0}
    df["Rock_Type_Code"] = df["Rock_Type"].map(rock_map).fillna(0).astype(int)
    df["Trap_Type_Code"] = df["Trap_Type"].map(trap_map).fillna(0).astype(int)
    return df

def load_equipment_failure():
    path = DATA_DIR / "ai4i2020.csv"
    df = pd.read_csv(path)
    df = df.rename(columns={
        "Air temperature [K]":     "Air_Temp_K",
        "Process temperature [K]": "Process_Temp_K",
        "Rotational speed [rpm]":  "Rot_Speed_rpm",
        "Torque [Nm]":             "Torque_Nm",
        "Tool wear [min]":         "Tool_Wear_min",
        "Machine failure":         "Machine_Failure",
    })
    df["Temp_Diff"] = df["Process_Temp_K"] - df["Air_Temp_K"]
    df["Power"]     = df["Torque_Nm"] * df["Rot_Speed_rpm"] / 9550
    type_map = {"L": 0, "M": 1, "H": 2}
    df["Type_Code"] = df["Type"].map(type_map).fillna(1).astype(int)
    return df

def load_energy_data():
    path = DATA_DIR / "Energy___Fuel_Consumption_Data.csv"
    df_raw = pd.read_excel(path, header=None)
    years = list(range(2008, 2023))
    records = []
    for i, row in df_raw.iterrows():
        energy_type = str(row.iloc[1]).strip()
        values = row.iloc[2:].tolist()
        for j, val in enumerate(values):
            year = years[j] if j < len(years) else 2008 + j
            numeric_val = pd.to_numeric(str(val).replace("-", "").strip(), errors="coerce")
            records.append({
                "Country":     "Egypt",
                "Energy_Type": energy_type,
                "Year":        year,
                "Consumption": float(numeric_val) if not pd.isna(numeric_val) else 0.0,
            })
    return pd.DataFrame(records)

def get_well_summary(df_drilling):
    agg_dict = {
        "Depth_Max":  ("DEPTH_MD", "max"),
        "Depth_Min":  ("DEPTH_MD", "min"),
        "Avg_GR":     ("GR",       "mean"),
        "Avg_RHOB":   ("RHOB",     "mean"),
        "Avg_NPHI":   ("NPHI",     "mean"),
        "Formations": ("FORMATION","nunique"),
    }
    # Add ROP and RDEP only if they exist (some wells may not have them)
    if "ROP" in df_drilling.columns:
        agg_dict["Avg_ROP"] = ("ROP", "mean")
    if "RDEP" in df_drilling.columns:
        agg_dict["Avg_RDEP"] = ("RDEP", "mean")
    return df_drilling.groupby("WELL").agg(**agg_dict).reset_index()

if __name__ == "__main__":
    print("Loading all datasets...")
    d = load_drilling_sensor()
    o = load_oil_well_log()
    e = load_equipment_failure()
    f = load_energy_data()
    print(f"Drilling sensor:   {d.shape}  |  Wells: {d['WELL'].nunique()}")
    print(f"Oil well log:      {o.shape}")
    print(f"Equipment failure: {e.shape}")
    print(f"Energy data:       {f.shape}")
    print("All datasets loaded successfully.")