"""Prints test-set performance from data/test_predictions.csv (written by the Kaggle notebook).
Uses only pandas/numpy. Run:  python check_results.py
"""
import json
import os

import numpy as np
import pandas as pd

df = pd.read_csv("data/test_predictions.csv")
thr = json.load(open("models/model_config.json"))["threshold"] if os.path.exists("models/model_config.json") else 0.5
y = df["label"].to_numpy()                 # 1 = leukemia (all), 0 = normal (hem)
p = df["prob_leukemia"].to_numpy()
n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())


def auc(y, p):                              # rank-based ROC-AUC
    r = pd.Series(p).rank(method="average").to_numpy()
    return (r[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def metrics(t):
    pred = (p >= t).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    d = lambda a, b: a / b if b else float("nan")
    return dict(tp=tp, fn=fn, fp=fp, tn=tn, sens=d(tp, tp + fn), spec=d(tn, tn + fp),
                ppv=d(tp, tp + fp), npv=d(tn, tn + fn), acc=d(tp + tn, len(y)), flagged=d(tp + fp, len(y)))


print(f"Test images: {len(df)} | patients: {df['patient'].nunique() if 'patient' in df else 'n/a'}")
print(f"Leukemia: {n_pos} ({n_pos/len(df):.0%}) | Normal: {n_neg} ({n_neg/len(df):.0%})")
print(f"ROC-AUC: {auc(y, p):.3f}\n")

m = metrics(thr)
print(f"At your saved threshold {thr:.2f}:")
print(f"                 predicted Normal   predicted Leukemia")
print(f"  true Normal      {m['tn']:8d}           {m['fp']:8d}")
print(f"  true Leukemia    {m['fn']:8d}           {m['tp']:8d}")
print(f"  sensitivity (leukemia caught): {m['sens']:.1%}")
print(f"  specificity (normal cleared):  {m['spec']:.1%}")
print(f"  precision (flags that are real): {m['ppv']:.1%} | NPV: {m['npv']:.1%} | accuracy: {m['acc']:.1%}")
print(f"  share of images flagged: {m['flagged']:.0%}\n")

print("Threshold sweep (sensitivity vs specificity):")
print("  thr   sens   spec   flagged")
for t in [0.2, 0.3, 0.4, 0.5, 0.6, thr, 0.7, 0.8, 0.9]:
    r = metrics(t)
    star = "  <- saved" if t == thr else ""
    print(f"  {t:.2f}  {r['sens']:.1%}  {r['spec']:.1%}  {r['flagged']:.0%}{star}")

# ---- per-patient view: are the errors spread out, or concentrated in a few patients?
if "patient" in df.columns:
    g = df.assign(flag=(p >= thr).astype(int)).groupby("patient").agg(
        images=("label", "size"), true_leukemia=("label", "mean"),
        mean_prob=("prob_leukemia", "mean"), flagged_share=("flag", "mean"))
    g["true_class"] = np.where(g["true_leukemia"] >= 0.5, "Leukemia", "Normal")
    g = g.sort_values(["true_class", "flagged_share"])
    print("\nPer patient (flagged_share = share of that patient's cells flagged leukemia):")
    print(f"  {'patient':10s} {'true':9s} {'cells':>5s} {'mean P':>7s} {'flagged':>8s}")
    for pid, r in g.iterrows():
        print(f"  {pid:10s} {r['true_class']:9s} {int(r['images']):5d} {r['mean_prob']:7.2f} {r['flagged_share']:8.0%}")
    mixed = g[(g["true_leukemia"] > 0) & (g["true_leukemia"] < 1)]
    if len(mixed):
        print(f"  WARNING: {len(mixed)} patient(s) have cells in both classes. Check the patient-ID parsing.")
