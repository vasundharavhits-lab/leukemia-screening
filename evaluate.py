"""Batch test: run labelled images through the same pipeline the app uses.

Folder layout (folder name = true label):
    my_test_images/
        Benign/   a.jpg b.jpg ...
        Early/    ...
        Pre/      ...
        Pro/      ...
Single-cell images work too, e.g. folders named Normal and Leukemia.

Folders named Benign, Normal, HEM or Negative count as "not leukemia".
Every other folder name counts as "leukemia".

Run from the project folder (the one that contains app.py and src/):
    python evaluate.py my_test_images
Optional:  python evaluate.py my_test_images --out results.csv
"""
import argparse
import os
import sys

import pandas as pd

from src.features import decode_image
from src.model import LeukemiaModel

MODEL_PATH = os.getenv("MODEL_PATH", "models/final_mobilenetv2_finetuned.keras")
CONFIG_PATH = os.getenv("CONFIG_PATH", "models/model_config_mnv2.json")
NEGATIVE = {"benign", "normal", "hem", "negative"}
EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--out", default="evaluation_results.csv")
    args = ap.parse_args()

    if not os.path.isdir(args.folder):
        sys.exit(f"Folder not found: {args.folder}")

    model = LeukemiaModel(MODEL_PATH, CONFIG_PATH)
    rows = []
    classes = sorted(d for d in os.listdir(args.folder)
                     if os.path.isdir(os.path.join(args.folder, d)))
    if not classes:
        sys.exit("No class sub-folders found. See the layout at the top of this file.")

    for cls in classes:
        folder = os.path.join(args.folder, cls)
        files = sorted(f for f in os.listdir(folder) if f.lower().endswith(EXTS))
        true_pos = cls.strip().lower() not in NEGATIVE
        print(f"{cls}: {len(files)} images ({'leukemia' if true_pos else 'not leukemia'})")
        for i, fn in enumerate(files, 1):
            row = {"class": cls, "file": fn, "true_leukemia": true_pos,
                   "route": None, "p_leukemia": None, "pred_leukemia": None,
                   "status": "ok", "note": ""}
            try:
                bgr = decode_image(open(os.path.join(folder, fn), "rb").read())
                p = model.predict(bgr)
                row.update(route=model.last_route, p_leukemia=p,
                           pred_leukemia=bool(p >= model.threshold))
            except ValueError as e:               # rejected by the input check, or unreadable
                row.update(status="rejected", note=str(e))
            except Exception as e:
                row.update(status="error", note=str(e))
            rows.append(row)
            if i % 50 == 0:
                print(f"  {cls}: {i}/{len(files)}")

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)

    n = len(df)
    ok = df[df["status"] == "ok"]
    print("\n================ RESULTS ================")
    print(f"Images: {n} | analyzed: {len(ok)} | rejected: {(df.status == 'rejected').sum()} "
          f"| errors: {(df.status == 'error').sum()}")
    if len(ok):
        tp = int(((ok.true_leukemia) & (ok.pred_leukemia)).sum())
        fn = int(((ok.true_leukemia) & (~ok.pred_leukemia)).sum())
        tn = int(((~ok.true_leukemia) & (~ok.pred_leukemia)).sum())
        fp = int(((~ok.true_leukemia) & (ok.pred_leukemia)).sum())
        sens = tp / (tp + fn) if (tp + fn) else float("nan")
        spec = tn / (tn + fp) if (tn + fp) else float("nan")
        acc = (tp + tn) / len(ok)
        print(f"Leukemia caught (sensitivity): {sens:.1%}  ({tp} of {tp + fn})")
        print(f"Not-leukemia correct (specificity): {spec:.1%}  ({tn} of {tn + fp})")
        print(f"Accuracy on analyzed images: {acc:.1%}")
        print(f"Missed leukemia (false negatives): {fn} | False alarms (false positives): {fp}")
        print("\nConfusion matrix (rows = true, columns = predicted):")
        print(pd.crosstab(ok["class"], ok["pred_leukemia"].map(
            {True: "pred leukemia", False: "pred not leukemia"})))
        print("\nBy model route:")
        print(ok.groupby("route").size().to_string())

    rej = df[df.status != "ok"]
    if len(rej):
        print(f"\nRejected or failed images by class (the input check said no):")
        print(rej.groupby(["class", "status"]).size().to_string())
    wrong = ok[ok.true_leukemia != ok.pred_leukemia] if len(ok) else ok
    if len(wrong):
        print("\nWrong predictions (check these images by eye):")
        print(wrong[["class", "file", "route", "p_leukemia"]].to_string(index=False))
    print(f"\nFull per-image results saved to {args.out}")


if __name__ == "__main__":
    main()
