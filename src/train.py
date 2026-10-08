"""
Train a Random Forest flood-susceptibility model on data/samples.csv.

- 5-fold stratified cross-validation (repeated 3x) for stable estimates
- Final model trained on all data; metrics also reported on a 25% held-out split
- Outputs: outputs/metrics.json, outputs/feature_importance.png,
  outputs/susceptibility.tif (full-grid RF probability), outputs/susceptibility.png
"""
import json
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, precision_score, recall_score,
                             f1_score, confusion_matrix)
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split

SRC_DIR = os.path.dirname(__file__)
DATA = os.path.join(SRC_DIR, "..", "data")
OUT = os.path.join(SRC_DIR, "..", "outputs")
SEED = 42

FEATURES = ["elevation", "slope_deg", "log_flowacc", "twi",
            "dist_waterway_m", "dist_road_m"]


def metrics_at(y_true, proba, thresh=0.5):
    pred = (proba >= thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    return {
        "auc": float(roc_auc_score(y_true, proba)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def main():
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_csv(os.path.join(DATA, "samples.csv"))
    X = df[FEATURES].values.astype(np.float32)
    y = df["label"].values.astype(int)
    print(f"samples: {len(df)}  positives: {y.sum()}  negatives: {(y==0).sum()}")

    rf = RandomForestClassifier(n_estimators=300, max_depth=None,
                                min_samples_leaf=2, n_jobs=-1,
                                random_state=SEED, class_weight="balanced")

    # repeated stratified CV
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)
    aucs, f1s = [], []
    for tr, te in cv.split(X, y):
        rf.fit(X[tr], y[tr])
        p = rf.predict_proba(X[te])[:, 1]
        aucs.append(roc_auc_score(y[te], p))
        f1s.append(f1_score(y[te], (p >= 0.5).astype(int), zero_division=0))
    print(f"CV AUC: {np.mean(aucs):.3f} ± {np.std(aucs):.3f} | "
          f"CV F1: {np.mean(f1s):.3f} ± {np.std(f1s):.3f}")

    # held-out test split
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25,
                                          stratify=y, random_state=SEED)
    rf.fit(Xtr, ytr)
    heldout = metrics_at(yte, rf.predict_proba(Xte)[:, 1])
    print("held-out:", {k: round(v, 3) for k, v in heldout.items() if isinstance(v, float)})

    # final model on all data
    final = RandomForestClassifier(n_estimators=500, min_samples_leaf=2,
                                   n_jobs=-1, random_state=SEED,
                                   class_weight="balanced")
    final.fit(X, y)
    joblib.dump(final, os.path.join(OUT, "rf_model.joblib"))
    importances = dict(zip(FEATURES, final.feature_importances_.tolist()))

    metrics = {
        "n_samples": int(len(df)),
        "n_positive": int(y.sum()),
        "n_negative": int((y == 0).sum()),
        "cv_auc_mean": float(np.mean(aucs)),
        "cv_auc_std": float(np.std(aucs)),
        "cv_f1_mean": float(np.mean(f1s)),
        "cv_f1_std": float(np.std(f1s)),
        "heldout_test": heldout,
        "feature_importances": importances,
        "model": "RandomForestClassifier(n_estimators=500, class_weight=balanced)",
        "seed": SEED,
    }
    with open(os.path.join(OUT, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print("wrote outputs/metrics.json")

    # feature importance chart
    order = np.argsort(final.feature_importances_)
    plt.figure(figsize=(7, 4))
    plt.barh([FEATURES[i] for i in order], final.feature_importances_[order])
    plt.xlabel("Gini importance")
    plt.title("Flood susceptibility — feature importance")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "feature_importance.png"), dpi=150)
    plt.close()

    # ---- full-grid prediction ----
    print("predicting full grid (chunked)...")
    stack = []
    with rasterio.open(os.path.join(DATA, "derived", "elevation.tif")) as src:
        profile = src.profile.copy()
        shape = (src.height, src.width)
    for name in FEATURES:
        with rasterio.open(os.path.join(DATA, "derived", f"{name}.tif")) as src:
            stack.append(src.read(1).astype(np.float32))
    Xg = np.stack(stack, axis=-1)  # h x w x 6
    h, w = Xg.shape[:2]
    proba = np.full((h, w), np.nan, dtype=np.float32)
    chunk = 60000
    flat = Xg.reshape(-1, len(FEATURES))
    valid = np.isfinite(flat).all(axis=1) & (flat[:, 0] > 1.0)
    idx = np.where(valid)[0]
    for s in range(0, len(idx), chunk):
        sl = idx[s:s + chunk]
        proba.ravel()[sl] = final.predict_proba(flat[sl])[:, 1]
        if s % (chunk * 5) == 0:
            print(f"  {s}/{len(idx)}", end="\r")

    profile.update(dtype="float32", count=1, compress="deflate", nodata=-9999)
    with rasterio.open(os.path.join(OUT, "susceptibility.tif"), "w", **profile) as dst:
        dst.write(np.nan_to_num(proba, nan=-9999).astype(np.float32), 1)
    print("\nwrote outputs/susceptibility.tif")

    # PNG visualization
    import geopandas as gpd
    fig, ax = plt.subplots(figsize=(10, 7))
    disp = np.ma.masked_invalid(proba)
    im = ax.imshow(disp, cmap="YlOrRd", vmin=0, vmax=1, origin="upper",
                   extent=[profile["transform"].c,
                           profile["transform"].c + w * profile["transform"].a,
                           profile["transform"].f + h * profile["transform"].e,
                           profile["transform"].f])
    # overlay hotspots
    hs = pd.read_csv(os.path.join(DATA, "label_provenance.csv"))
    from rasterio.warp import transform as warp_transform
    hx, hy = warp_transform("EPSG:4326", "EPSG:32630",
                            hs["lon"].tolist(), hs["lat"].tolist())
    ax.scatter(hx, hy, c="black", s=18, marker="x", label="documented hotspots",
               linewidths=1.2)
    ax.set_xlabel("Easting (m, EPSG:32630)")
    ax.set_ylabel("Northing (m, EPSG:32630)")
    ax.set_title("Flood susceptibility — Greater Accra (Random Forest)")
    fig.colorbar(im, ax=ax, label="P(flood-prone)")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "susceptibility.png"), dpi=150)
    plt.close()
    print("wrote outputs/susceptibility.png")


if __name__ == "__main__":
    main()
