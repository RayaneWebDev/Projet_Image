"""
Construction des bases d'exemples pour les deux pipelines de classification.

Pour chaque annotation JSON, on recherche la pièce détectée la plus proche
(distance centre-centre < 800 px), on extrait son vecteur ring_features
et on l'associe au label annoté.

Produit deux fichiers :
  - model/knn_database.npy     : base pour le pipeline k-NN classique
  - model/knn_database_rf.npy  : base pour le pipeline ExtraTrees (RF)

Utilisation :
    python training/build_knn_profiles.py

Auteurs : Équipe ImageGroupe
Date    : 2026
"""
import cv2
import numpy as np
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.segmentation import segment_piece
from core.features import extract_features

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANNOT_DIR = os.path.join(BASE_DIR, "annotation")
IMG_DIR   = os.path.join(BASE_DIR, "data", "validation")
OUT_PATH  = os.path.join(BASE_DIR, "model", "knn_database.npy")
RF_OUT_PATH = os.path.join(BASE_DIR, "model", "knn_database_rf.npy")

# Distance maximale (px) entre le centre annoté et le centre détecté
# pour qu'une détection soit associée à une annotation
DIST_THR = 800

# Normalisation des variantes orthographiques vers le format interne du pipeline
LABEL_MAP = {
    "1cent":   "1 cent",  "2cents":  "2 cent",  "5cents":  "5 cent",
    "10cents": "10 cent", "20cents": "20 cent",  "50cents": "50 cent",
    "1euro":   "1 Euro",  "2euros":  "2 Euro",  "1 euro":  "1 Euro",
    "2euro":   "2 Euro",  "10cent":  "10 cent", "20cent":  "20 cent",
    "50cent":  "50 cent", "1 cent":  "1 cent",  "2 cent":  "2 cent",
    "5 cent":  "5 cent",  "10 cent": "10 cent", "20 cent": "20 cent",
    "50 cent": "50 cent", "1 Euro":  "1 Euro",  "2 Euro":  "2 Euro",
}

os.makedirs(os.path.join(BASE_DIR, "model"), exist_ok=True)


# ── Base k-NN classique (knn_database.npy) ────────────────────────────────────
print("--- Construction de la base k-NN (knn_database.npy) ---")

X = []  # vecteurs ring_features bruts
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

    # Redimensionnement si l'image dépasse 1920 px sur le grand côté
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

        # Recherche de la détection la plus proche de l'annotation
        best_feat, best_dist = None, float("inf")
        for feat in feats:
            fx, fy = feat["center"]
            dist   = np.sqrt((fx - cx_gt) ** 2 + (fy - cy_gt) ** 2)
            if dist < best_dist:
                best_dist = dist
                best_feat = feat

        if best_feat is not None and best_dist < DIST_THR:
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


# ── Base ExtraTrees (knn_database_rf.npy) ─────────────────────────────────────
# Même logique que la base classique ; les features sont aplaties en float32
# pour être directement compatibles avec ExtraTreesClassifier.
print("\n--- Construction de la base RF (knn_database_rf.npy) ---")

X_rf   = []
y_rf   = []
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
            dist   = np.sqrt((fx - cx_gt) ** 2 + (fy - cy_gt) ** 2)
            if dist < best_dist:
                best_dist = dist
                best_feat = feat

        if best_feat is not None and best_dist < DIST_THR:
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
