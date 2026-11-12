"""Builds exports/demo_scans.csv for DESIGNING the Power BI report.

It reuses the real model scores from Kaggle's test_predictions.csv, but the dates, ages,
genders, labs and review statuses are RANDOM placeholders. This file is demo data only:
it is never written to scans.db, and it must not be used for any real reporting.
The columns match the real scan log, so later you only change the file/source in Power BI.
"""
import json
import os

import numpy as np
import pandas as pd

SRC = "data/test_predictions.csv"
CONFIG = "models/model_config.json"
OUT = "exports/demo_scans.csv"
N = 120
rng = np.random.default_rng(7)

df = pd.read_csv(SRC)
threshold = json.load(open(CONFIG))["threshold"] if os.path.exists(CONFIG) else 0.5
df = df.sample(min(N, len(df)), random_state=7).reset_index(drop=True)

days = rng.integers(0, 14, len(df))                       # spread over the last 14 days
secs = rng.integers(8 * 3600, 19 * 3600, len(df))         # working hours
when = pd.Timestamp.now().normalize() - pd.to_timedelta(days, unit="D") + pd.to_timedelta(secs, unit="s")

prob = df["prob_leukemia"].astype(float)
out = pd.DataFrame({
    "id": np.arange(1, len(df) + 1),
    "scanned_at": when,
    "patient_id": df["patient"],
    "age": rng.integers(3, 70, len(df)),
    "gender": rng.choice(["Male", "Female"], len(df)),
    "lab": rng.choice(["Indore Lab", "Bhopal Lab", "Mumbai Lab"], len(df), p=[0.5, 0.3, 0.2]),
    "image_name": df["path"].str.split("/").str[-1],
    "prob_leukemia": prob.round(4),
    "threshold": threshold,
    "prediction": np.where(prob >= threshold, "Leukemia (ALL)", "Normal (HEM)"),
    "uncertain": ((prob - threshold).abs() < 0.10).astype(int),
    "nucleus_area": np.nan, "nucleus_circularity": np.nan,
    "nucleus_solidity": np.nan, "nc_ratio": np.nan,
    "review_status": rng.choice(["Pending", "Confirmed", "Rejected", "Needs second opinion"],
                                len(df), p=[0.4, 0.45, 0.1, 0.05]),
}).sort_values("scanned_at").reset_index(drop=True)
out["id"] = np.arange(1, len(out) + 1)

os.makedirs("exports", exist_ok=True)
out.to_csv(OUT, index=False)
print(f"Wrote {OUT}: {len(out)} rows, {int((out.prediction.str.startswith('Leukemia')).sum())} flagged leukemia")
