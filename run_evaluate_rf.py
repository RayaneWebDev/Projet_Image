"""
Evaluation du pipeline avec classification Random Forest.
Comparer avec run_evaluate.py (k-NN).
"""
import os
import json
import cv2

from core.segmentation import segment_piece
from core.features import extract_features
from core.classification_ml import classify_piece_rf
from evaluation.metrics import compute_metrics
from evaluation.evaluate import load_annotation

IOU_THRESHOLD     = 0.3
SUCCESS_PRECISION = 0.75
SUCCESS_RECALL    = 0.75
SUCCESS_LABEL_ACC = 0.70

VAL_DIR = "data/validation"
ANN_DIR = "annotation"

image_ext = {".jpg", ".jpeg", ".png", ".bmp"}
files     = sorted([f for f in os.listdir(VAL_DIR)
                    if os.path.splitext(f)[1].lower() in image_ext])

all_tp = all_fp = all_fn = all_tp_correct = 0

for fname in files:
    img_path = os.path.join(VAL_DIR, fname)
    ann_path = os.path.join(ANN_DIR, os.path.splitext(fname)[0] + ".json")
    if not os.path.exists(ann_path):
        continue

    img = cv2.imread(img_path)
    if img is None:
        continue

    with open(ann_path) as f:
        ann_data = json.load(f)
    orig_w    = ann_data.get("imageWidth", img.shape[1])
    scale     = min(1.0, 1200 / max(img.shape[:2]))
    if scale < 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale)
    ann_scale = img.shape[1] / orig_w

    gt       = load_annotation(ann_path, ann_scale)
    circles  = segment_piece(img)
    feats, _ = extract_features(circles, img)

    detects = []
    for feat in feats:
        label, d_mm, conf = classify_piece_rf(feat, feats)
        cx, cy = feat["center"]
        r      = feat["radius"]
        detects.append((cx, cy, r, label))

    m = compute_metrics(detects, gt, IOU_THRESHOLD)
    all_tp         += m["TP"]
    all_fp         += m["FP"]
    all_fn         += m["FN"]
    all_tp_correct += sum(1 for t in m["tp_details"] if t["label_correct"])

total_pred = all_tp + all_fp
total_gt   = all_tp + all_fn
g_prec     = all_tp / total_pred if total_pred > 0 else 0.0
g_recall   = all_tp / total_gt   if total_gt   > 0 else 0.0
g_f1       = 2 * g_prec * g_recall / (g_prec + g_recall) if (g_prec + g_recall) > 0 else 0.0
g_lbl_acc  = all_tp_correct / all_tp if all_tp > 0 else 0.0
success    = (g_prec >= SUCCESS_PRECISION and
              g_recall >= SUCCESS_RECALL and
              g_lbl_acc >= SUCCESS_LABEL_ACC)

print(f"\n{'='*65}")
print(f"  RESULTATS RANDOM FOREST ({len(files)} images evaluees)")
print(f"{'='*65}")
print(f"  TP={all_tp}  FP={all_fp}  FN={all_fn}")
print(f"  Precision                : {g_prec:.3f}")
print(f"  Rappel                   : {g_recall:.3f}")
print(f"  F1-score                 : {g_f1:.3f}")
print(f"  Taux d'identification    : {g_lbl_acc:.3f}")
print(f"  VERDICT : {'VALIDATION REUSSIE' if success else 'VALIDATION ECHOUEE'}")
print(f"{'='*65}\n")