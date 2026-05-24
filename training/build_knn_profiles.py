"""
Construit la base d'exemples k-NN depuis data/validation et les annotations.
Pour chaque annotation, on cherche la piece detectee la plus proche,
on extrait son vecteur ring_features et on l'associe au label annote.

Deux bases :
  - model/knn_database.npy     (pipeline classique, inchangee)
  - model/knn_database_rf.npy  (Random Forest, seuil elargi + augmentation)
"""
import cv2
import numpy as np
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.segmentation import segment_piece
from core.features import extract_features
from core.utils import COIN_DIAMETERS_MM

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANNOT_DIR = os.path.join(BASE_DIR, "annotation")
IMG_DIR   = os.path.join(BASE_DIR, "data", "validation")
OUT_PATH  = os.path.join(BASE_DIR, "model", "knn_database.npy")

RF_OUT_PATH  = os.path.join(BASE_DIR, "model", "knn_database_rf.npy")
RF_DIST_THR  = 800

COLOR_CANDS_RF = {
    "bronze":  ["1 cent", "2 cent", "5 cent"],
    "gold":    ["10 cent", "20 cent", "50 cent"],
    "silver":  ["1 Euro", "2 Euro"],
    "unknown": list(COIN_DIAMETERS_MM.keys()),
}
D_MM_MIN = 16.0   # 1ct = 16.25mm
D_MM_MAX = 26.0   # 2eu = 25.75mm


def _compute_sf(feats):
    """Estime le scale factor global d'une image multi-pieces."""
    diam_px    = [f["diameter_pixels"] for f in feats]
    color_labs = [f.get("color_label", "unknown") for f in feats]
    best_sf, best_err = None, float("inf")
    for d, color in zip(diam_px, color_labs):
        for ref_label in COLOR_CANDS_RF.get(color, list(COIN_DIAMETERS_MM.keys())):
            sf        = COIN_DIAMETERS_MM[ref_label] / d
            total_err = sum(
                min(abs(COIN_DIAMETERS_MM[l] - d2 * sf) / COIN_DIAMETERS_MM[l]
                    for l in COLOR_CANDS_RF.get(c2, list(COIN_DIAMETERS_MM.keys())))
                for d2, c2 in zip(diam_px, color_labs)
            )
            if total_err < best_err:
                best_err = total_err
                best_sf  = sf
    return best_sf or 0.1


def _augment(X_base, y_base, n_copies, sigma, rng):
    """Genere n_copies copies bruitees de chaque exemple (meme label)."""
    X_aug, y_aug = [], []
    x_std = X_base.std(axis=0).mean() or 1.0
    noise_scale = sigma * x_std
    for x, label in zip(X_base, y_base):
        for _ in range(n_copies):
            noisy = x + rng.normal(0, noise_scale, size=x.shape)
            noisy = np.clip(noisy, 0, None).astype(np.float32)
            X_aug.append(noisy)
            y_aug.append(label)
    return np.array(X_aug, dtype=np.float32), np.array(y_aug)

# Normalisation des labels d'annotation vers le format pipeline
LABEL_MAP = {
    "1cent":  "1 cent",  "2cents": "2 cent",  "5cents": "5 cent",
    "10cents":"10 cent", "20cents":"20 cent",  "50cents":"50 cent",
    "1euro":  "1 Euro",  "2euros": "2 Euro",  "10cent": "10 cent",
    "20cent": "20 cent", "50cent": "50 cent", "1 cent": "1 cent",
    "2 cent": "2 cent",  "5 cent": "5 cent",  "10 cent":"10 cent",
    "20 cent":"20 cent", "50 cent":"50 cent",  "1 Euro": "1 Euro",
    "2 Euro": "2 Euro",  "1 euro": "1 Euro",  "2 euro": "2 Euro",
}

os.makedirs(os.path.join(BASE_DIR, "model"), exist_ok=True)

X = []  # vecteurs ring_features
y = []  # labels correspondants

total_found = 0
total_annot = 0

for json_file in sorted(os.listdir(ANNOT_DIR)):
    if not json_file.endswith(".json"):
        continue

    with open(os.path.join(ANNOT_DIR, json_file)) as f:
        data = json.load(f)

    img_name = data.get("imagePath", "").replace("\\", "/").split("/")[-1]
    img_path = os.path.join(IMG_DIR, img_name)
    if not os.path.exists(img_path):
        continue

    img = cv2.imread(img_path)
    if img is None:
        continue

    h, w = img.shape[:2]
    if max(h, w) > 1920:
        img = cv2.resize(img, None, fx=1920 / max(h, w), fy=1920 / max(h, w))

    circles = segment_piece(img)
    feats, _ = extract_features(circles, img)

    for shape in data.get("shapes", []):
        label_raw = shape.get("label", "").strip()
        label     = LABEL_MAP.get(label_raw, label_raw)
        points    = shape.get("points", [])
        if len(points) < 2:
            continue

        total_annot += 1
        xs    = [p[0] for p in points]
        ys    = [p[1] for p in points]
        cx_gt = (min(xs) + max(xs)) / 2
        cy_gt = (min(ys) + max(ys)) / 2

        # Piece detectee la plus proche de l'annotation
        best_feat, best_dist = None, float("inf")
        for feat in feats:
            fx, fy = feat["center"]
            dist   = np.sqrt((fx - cx_gt)**2 + (fy - cy_gt)**2)
            if dist < best_dist:
                best_dist = dist
                best_feat = feat

        if best_feat is not None and best_dist < 800:
            rf = best_feat.get("ring_features")
            if rf is not None and rf.sum() > 0:
                X.append(rf)
                y.append(label)
                total_found += 1

X = np.array(X)
y = np.array(y)

np.save(OUT_PATH, {"X": X, "y": y})

print(f"Annotations totales : {total_annot}")
print(f"Exemples captures   : {total_found}")
print()
for label in sorted(set(y)):
    print(f"  {label:10s} : {np.sum(y == label):3d} exemples")

# ── Base RF (ring_features 240-dim) ───────────────────────────────────
print("\n--- Construction de la base RF (knn_database_rf.npy) ---")
X_rf, y_rf = [], []
total_rf = 0

for json_file in sorted(os.listdir(ANNOT_DIR)):
    if not json_file.endswith(".json"):
        continue
    with open(os.path.join(ANNOT_DIR, json_file)) as f:
        data = json.load(f)
    img_name = data.get("imagePath", "").replace("\\", "/").split("/")[-1]
    img_path = os.path.join(IMG_DIR, img_name)
    if not os.path.exists(img_path):
        continue
    img = cv2.imread(img_path)
    if img is None:
        continue
    h, w = img.shape[:2]
    if max(h, w) > 1920:
        img = cv2.resize(img, None, fx=1920 / max(h, w), fy=1920 / max(h, w))
    circles = segment_piece(img)
    feats, _ = extract_features(circles, img)

    for shape in data.get("shapes", []):
        label_raw = shape.get("label", "").strip()
        label     = LABEL_MAP.get(label_raw, label_raw)
        points    = shape.get("points", [])
        if len(points) < 2:
            continue
        xs    = [p[0] for p in points]
        ys_   = [p[1] for p in points]
        cx_gt = (min(xs) + max(xs)) / 2
        cy_gt = (min(ys_) + max(ys_)) / 2

        best_feat, best_dist = None, float("inf")
        for feat in feats:
            fx, fy = feat["center"]
            dist   = np.sqrt((fx - cx_gt)**2 + (fy - cy_gt)**2)
            if dist < best_dist:
                best_dist = dist
                best_feat = feat

        if best_feat is not None and best_dist < RF_DIST_THR:
            rf_vec = best_feat.get("ring_features")
            if rf_vec is not None and rf_vec.sum() > 0:
                X_rf.append(rf_vec.flatten().astype(np.float32))
                y_rf.append(label)
                total_rf += 1

X_rf = np.array(X_rf, dtype=np.float32)
y_rf = np.array(y_rf)

np.save(RF_OUT_PATH, {"X": X_rf, "y": y_rf})

print(f"Exemples captures : {total_rf}  (features: {X_rf.shape[1]})")
print()
for label in sorted(set(y_rf)):
    print(f"  {label:10s} : {np.sum(y_rf == label):3d} exemples")