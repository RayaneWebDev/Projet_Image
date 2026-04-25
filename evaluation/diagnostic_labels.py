"""
Affiche la matrice de confusion et l'accuracy par classe
sur le jeu de validation.
"""
import os
import json
import cv2
from collections import defaultdict

from evaluation.evaluate import load_annotation, run_pipeline
from evaluation.metrics import compute_metrics, normalize_label

VAL_DIR = "data/validation"
ANN_DIR = "annotation"

confusion = defaultdict(lambda: defaultdict(int))
image_ext = {".jpg", ".jpeg", ".png", ".bmp"}

files = sorted([
    f for f in os.listdir(VAL_DIR)
    if os.path.splitext(f)[1].lower() in image_ext
])

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
    orig_w = ann_data.get("imageWidth", img.shape[1])

    scale = min(1.0, 1200 / max(img.shape[:2]))
    if scale < 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale)
    ann_scale = img.shape[1] / orig_w

    gt      = load_annotation(ann_path, ann_scale)
    detects = run_pipeline(img)
    dets    = [(cx, cy, r, lbl) for cx, cy, r, lbl, _, _ in detects]

    m = compute_metrics(dets, gt, iou_threshold=0.30)

    for t in m["tp_details"]:
        confusion[normalize_label(t["gt_label"])][normalize_label(t["det_label"])] += 1

# Matrice de confusion
print("\n=== MATRICE DE CONFUSION (GT -> PREDIT) ===\n")

all_labels = sorted(set(
    lbl
    for gt_lbl, preds in confusion.items()
    for lbl in [gt_lbl] + list(preds.keys())
))

print(f"{'GT \\ PRED':<12}", end="")
for lbl in all_labels:
    print(f"{lbl:>12}", end="")
print()

for gt_lbl in all_labels:
    if gt_lbl not in confusion:
        continue
    total   = sum(confusion[gt_lbl].values())
    correct = confusion[gt_lbl].get(gt_lbl, 0)
    print(f"{gt_lbl:<12}", end="")
    for pred_lbl in all_labels:
        count = confusion[gt_lbl].get(pred_lbl, 0)
        if count == 0:
            marker = "."
        elif gt_lbl == pred_lbl:
            marker = f"{count}OK"
        else:
            marker = f"{count}X"
        print(f"{marker:>12}", end="")
    acc = correct / total if total > 0 else 0.0
    print(f"   acc={acc:.2f}  ({correct}/{total})")

# Erreurs les plus frequentes
print("\n=== ERREURS LES PLUS FREQUENTES ===\n")
errors = [
    (count, gt_lbl, pred_lbl)
    for gt_lbl, preds in confusion.items()
    for pred_lbl, count in preds.items()
    if gt_lbl != pred_lbl
]
errors.sort(reverse=True)
for count, gt_lbl, pred_lbl in errors[:20]:
    print(f"  {count:3d}x  '{gt_lbl}' predit comme '{pred_lbl}'")

# Accuracy par classe
print("\n=== ACCURACY PAR CLASSE ===\n")
for gt_lbl in sorted(confusion.keys()):
    total   = sum(confusion[gt_lbl].values())
    correct = confusion[gt_lbl].get(gt_lbl, 0)
    acc     = correct / total if total > 0 else 0.0
    bar     = "#" * int(acc * 20) + "-" * (20 - int(acc * 20))
    print(f"  {gt_lbl:<10} [{bar}]  {acc:.2f}  ({correct}/{total})")

# Accuracy globale
total_correct = sum(confusion[l].get(l, 0) for l in confusion)
total_all     = sum(sum(preds.values()) for preds in confusion.values())
global_acc    = total_correct / total_all if total_all > 0 else 0.0
print(f"\n  Accuracy globale : {global_acc:.3f}  ({total_correct}/{total_all})")