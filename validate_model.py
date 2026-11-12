"""Validate the model on patient-separated data. Run ONCE per model version.

Input: validation_data.csv with columns: path, label, patient_id
  label: 0 = normal, 1 = leukemia
Output: validation_report.txt and train_patients.txt
"""
import os
import sys

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

MODEL_PATH = "models/final_mobilenetv2_finetuned.keras"
DATA_PATH = "validation_data.csv"
REPORT_PATH = "validation_report.txt"
TARGET_SENSITIVITY = 0.95


def leukemia_score(pred):
    # Current 4-class model: pred[0] is P(Benign). Change this if the model changes.
    return 1.0 - pred[0]


def load_scores(model, paths):
    imgs = []
    for p in paths:
        bgr = cv2.imread(p)
        if bgr is None:
            sys.exit(f"Could not read image: {p}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        imgs.append(cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA).astype("float32"))
    preds = model.predict(np.stack(imgs), verbose=0)
    return np.array([leukemia_score(p) for p in preds])


def split_by_patient(df):
    gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
    train_idx, rest_idx = next(gss.split(df, groups=df["patient_id"]))
    rest = df.iloc[rest_idx]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
    v_idx, t_idx = next(gss2.split(rest, groups=rest["patient_id"]))
    return df.iloc[train_idx], rest.iloc[v_idx], rest.iloc[t_idx]


def choose_threshold(y, s):
    best_t, best_spec = None, -1.0
    for t in np.unique(s):
        pred = s >= t
        sens = (pred & (y == 1)).sum() / max((y == 1).sum(), 1)
        spec = (~pred & (y == 0)).sum() / max((y == 0).sum(), 1)
        if sens >= TARGET_SENSITIVITY and spec > best_spec:
            best_t, best_spec = t, spec
    return best_t


def main():
    if os.path.exists(REPORT_PATH):
        sys.exit("validation_report.txt already exists. The test set may only be used once. "
                 "Delete the report only when starting a new model version.")

    df = pd.read_csv(DATA_PATH)
    train, val, test = split_by_patient(df)
    tr, va, te = set(train.patient_id), set(val.patient_id), set(test.patient_id)
    if (tr & va) or (tr & te) or (va & te):
        sys.exit("Patient overlap between splits. Check the patient IDs in the CSV.")

    train.patient_id.drop_duplicates().to_csv("train_patients.txt", index=False)
    print(f"Rows: train {len(train)}, val {len(val)}, test {len(test)}")
    print("Train on the patients in train_patients.txt only, then run this script again.")

    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    y_val = val.label.values
    s_val = load_scores(model, val.path)
    t = choose_threshold(y_val, s_val)

    if t is None:
        result = f"No threshold reaches {TARGET_SENSITIVITY:.0%} sensitivity on validation data. Model not usable."
        lines = [result]
    else:
        y_test = test.label.values
        s_test = load_scores(model, test.path)
        pred = s_test >= t
        sens = (pred & (y_test == 1)).sum() / (y_test == 1).sum()
        spec = (~pred & (y_test == 0)).sum() / (y_test == 0).sum()
        lines = [
            f"Model: {MODEL_PATH}",
            f"Threshold chosen on validation: {t:.4f}",
            f"Test rows: {len(test)} from {len(te)} patients",
            f"Test sensitivity: {sens:.3f}",
            f"Test specificity: {spec:.3f}",
            f"Test AUC: {roc_auc_score(y_test, s_test):.3f}",
        ]

    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"Saved {REPORT_PATH}")


if __name__ == "__main__":
    main()
