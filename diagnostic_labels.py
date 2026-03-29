# diagnostic_labels.py
import os, json
from collections import defaultdict
from evaluate import load_annotation, run_pipeline
from metrics import compute_metrics, normalize_label
import cv2

val_dir = "data/validation"
ann_dir = "annotation"

confusion = defaultdict(lambda: defaultdict(int))
image_ext = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG"}

files = sorted([f for f in os.listdir(val_dir)
                if os.path.splitext(f)[1].lower() in image_ext])

for fname in files:
    img_path = os.path.join(val_dir, fname)
    base     = os.path.splitext(fname)[0]
    ann_path = os.path.join(ann_dir, base + ".json")
    if not os.path.exists(ann_path):
        continue

    img = cv2.imread(img_path)
    if img is None:
        continue

    with open(ann_path) as f:
        ann_data = json.load(f)
    orig_w = ann_data.get("imageWidth",  img.shape[1])
    orig_h = ann_data.get("imageHeight", img.shape[0])

    scale = min(1.0, 1200 / max(img.shape[:2]))
    if scale < 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale)
    ann_scale = img.shape[1] / orig_w

    gt      = load_annotation(ann_path, ann_scale)
    detects = run_pipeline(img)

    # Adapter au nouveau format (cx, cy, r, label, d_mm, conf)
    dets = [(cx, cy, r, lbl) for cx, cy, r, lbl, d_mm, conf in detects]

    m = compute_metrics(dets, gt, iou_threshold=0.30)

    for t in m["tp_details"]:
        gt_lbl   = normalize_label(t["gt_label"])
        pred_lbl = normalize_label(t["det_label"])
        confusion[gt_lbl][pred_lbl] += 1

# ── Matrice de confusion ─────────────────────────────────────────
print("\n=== MATRICE DE CONFUSION (GT → PRÉDIT) ===\n")

all_labels = sorted(set(
    lbl
    for gt_lbl, preds in confusion.items()
    for lbl in [gt_lbl] + list(preds.keys())
))

# Header
header = "GT \\ PRED"
print(f"{header:<12}", end="")
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
            marker = f"{count}✓"
        else:
            marker = f"{count}✗"
        print(f"{marker:>12}", end="")
    acc = correct / total if total > 0 else 0.0
    print(f"   acc={acc:.2f}  ({correct}/{total})")

# ── Erreurs les plus fréquentes ──────────────────────────────────
print("\n=== ERREURS LES PLUS FRÉQUENTES ===\n")
errors = [
    (count, gt_lbl, pred_lbl)
    for gt_lbl, preds in confusion.items()
    for pred_lbl, count in preds.items()
    if gt_lbl != pred_lbl
]
errors.sort(reverse=True)
for count, gt_lbl, pred_lbl in errors[:20]:
    print(f"  {count:3d}x  '{gt_lbl}' prédit comme '{pred_lbl}'")

# ── Accuracy par classe ──────────────────────────────────────────
print("\n=== ACCURACY PAR CLASSE ===\n")
for gt_lbl in sorted(confusion.keys()):
    total   = sum(confusion[gt_lbl].values())
    correct = confusion[gt_lbl].get(gt_lbl, 0)
    acc     = correct / total if total > 0 else 0.0
    bar     = "█" * int(acc * 20) + "░" * (20 - int(acc * 20))
    print(f"  {gt_lbl:<10} {bar}  {acc:.2f}  ({correct}/{total})")

# ── Accuracy globale ─────────────────────────────────────────────
total_correct = sum(confusion[l].get(l, 0) for l in confusion)
total_all     = sum(sum(preds.values()) for preds in confusion.values())
global_acc    = total_correct / total_all if total_all > 0 else 0.0
print(f"\n  Accuracy globale : {global_acc:.3f}  ({total_correct}/{total_all})")