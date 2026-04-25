"""
Construit la base d'exemples k-NN depuis data/validation et les annotations.
Pour chaque annotation, on cherche la piece detectee la plus proche,
on extrait son vecteur ring_features et on l'associe au label annote.
Sauvegarde dans model/knn_database.npy
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

        if best_feat is not None and best_dist < 500:
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