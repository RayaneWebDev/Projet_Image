import os
import json
import cv2
import numpy as np
import argparse

from segmentation import segment_piece
from features import extract_features
from classification import classify_piece, get_coin_value
from utils import draw_label, COIN_VALUES_EUR
from metrics import compute_metrics, bbox_to_circle, normalize_label

# ── Seuils de réussite (modifiables) ─────────────────────────────
IOU_THRESHOLD     = 0.3
SUCCESS_PRECISION = 0.75
SUCCESS_RECALL    = 0.75
SUCCESS_LABEL_ACC = 0.70
# ─────────────────────────────────────────────────────────────────


def load_annotation(json_path, scale=1.0):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Les annotations ont été faites avec un rayon 2x trop petit
    # (annotateur a cliqué sur le bord intérieur de la pièce)
    ANNOTATION_RADIUS_FIX = 2.0

    ground_truths = []
    for shape in data.get("shapes", []):
        if shape.get("shape_type") != "circle":
            continue
        pts = shape["points"]
        if len(pts) < 2:
            continue
        cx, cy, r = bbox_to_circle(pts[0], pts[1])
        ground_truths.append((
            cx * scale,
            cy * scale,
            r  * scale * ANNOTATION_RADIUS_FIX,
            shape["label"]
        ))
    return ground_truths


def run_pipeline(image):
    """
    Exécute segmentation + features + classification sur une image.
    Retourne liste de (cx, cy, r, label, d_mm, conf).
    """
    circles = segment_piece(image)
    if not circles:
        return []

    features_list, _ = extract_features(circles, image)

    results = []
    for feat in features_list:
        label, d_mm, conf = classify_piece(feat, features_list)
        cx, cy = feat["center"]
        r      = feat["radius"]
        results.append((cx, cy, r, label, d_mm, conf))

    return results


def evaluate_dataset(val_dir, ann_dir,
                     iou_threshold=IOU_THRESHOLD,
                     save_dir=None):
    """
    Évalue le pipeline sur tout le dossier val_dir.
    Affiche les métriques par image et globales.
    Retourne True si la validation est réussie.
    """
    image_ext = {".jpg", ".jpeg", ".png", ".bmp"}
    files = sorted([
        f for f in os.listdir(val_dir)
        if os.path.splitext(f)[1].lower() in image_ext
    ])

    if not files:
        print(f"Aucune image dans {val_dir}")
        return False

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    all_tp = all_fp = all_fn = all_tp_correct = 0
    per_image = []

    print(f"\n{'='*65}")
    print(f"  VALIDATION — {len(files)} images  |  IoU≥{iou_threshold}"
          f"  P≥{SUCCESS_PRECISION}  R≥{SUCCESS_RECALL}"
          f"  LabelAcc≥{SUCCESS_LABEL_ACC}")
    print(f"{'='*65}")

    for fname in files:
        img_path = os.path.join(val_dir, fname)
        base     = os.path.splitext(fname)[0]
        ann_path = os.path.join(ann_dir, base + ".json")

        if not os.path.exists(ann_path):
            print(f"  [SKIP] {fname} — pas d'annotation")
            continue
        
        img = cv2.imread(img_path)
        if img is None:
            print(f"  [SKIP] {fname} — illisible")
            continue
        
        # Lire les dimensions originales depuis le JSON
        with open(ann_path, "r", encoding="utf-8") as f:
            ann_data = json.load(f)
        orig_w = ann_data.get("imageWidth",  img.shape[1])
        orig_h = ann_data.get("imageHeight", img.shape[0])

        # Redimensionner l'image
        max_side = max(img.shape[1], img.shape[0])
        scale = min(1.0, 1200 / max_side)
        if scale < 1.0:
            img = cv2.resize(img, None, fx=scale, fy=scale)

        # Scale des annotations basé sur les dimensions ORIGINALES du JSON
        ann_scale = img.shape[1] / orig_w   # scale réel image affichée / image annotée

        gt      = load_annotation(ann_path, ann_scale)
        detects = run_pipeline(img)
        dets    = [(cx, cy, r, lbl) for cx, cy, r, lbl, d_mm, _ in detects]

        m = compute_metrics(dets, gt, iou_threshold)
        all_tp         += m["TP"]
        all_fp         += m["FP"]
        all_fn         += m["FN"]
        all_tp_correct += sum(1 for t in m["tp_details"] if t["label_correct"])

        ok  = m["precision"] >= SUCCESS_PRECISION and \
              m["recall"]    >= SUCCESS_RECALL
        tag = "✓" if ok else "✗"

        print(f"\n  [{tag}] {fname}")
        print(f"       GT={len(gt)}  Dét={len(dets)}"
              f"  TP={m['TP']} FP={m['FP']} FN={m['FN']}"
              f"  P={m['precision']:.2f}  R={m['recall']:.2f}"
              f"  LblAcc={m['label_accuracy']:.2f}")

        for t in m["tp_details"]:
            if not t["label_correct"]:
                print(f"       ⚠ Label : prédit '{t['det_label']}'"
                      f" → attendu '{t['gt_label']}'  IoU={t['iou']:.2f}")
        for fp in m["fp_details"]:
            print(f"       FP  : '{fp['det_label']}'  IoU={fp['iou']:.2f}")
        for fn in m["fn_details"]:
            print(f"       FN  : '{fn['gt_label']}' non détecté")

        per_image.append({"file": fname, **m})

        # Sauvegarder image annotée
        if save_dir:
            _save_annotated(img, detects, gt,
                            os.path.join(save_dir, fname))

    # ── Résultats globaux ────────────────────────────────────────
    total_pred = all_tp + all_fp
    total_gt   = all_tp + all_fn

    g_prec     = all_tp / total_pred if total_pred > 0 else 0.0
    g_recall   = all_tp / total_gt   if total_gt   > 0 else 0.0
    g_f1       = (2 * g_prec * g_recall / (g_prec + g_recall)
                  if (g_prec + g_recall) > 0 else 0.0)
    g_lbl_acc  = all_tp_correct / all_tp if all_tp > 0 else 0.0

    success = (g_prec    >= SUCCESS_PRECISION and
               g_recall  >= SUCCESS_RECALL    and
               g_lbl_acc >= SUCCESS_LABEL_ACC)

    print(f"\n{'='*65}")
    print(f"  RÉSULTATS GLOBAUX ({len(per_image)} images évaluées)")
    print(f"{'='*65}")
    print(f"  TP={all_tp}  FP={all_fp}  FN={all_fn}")
    _print_metric("Précision",      g_prec,    SUCCESS_PRECISION)
    _print_metric("Rappel",         g_recall,  SUCCESS_RECALL)
    _print_metric("F1-score",       g_f1,      None)
    _print_metric("Label accuracy", g_lbl_acc, SUCCESS_LABEL_ACC)
    print(f"\n  VERDICT : {'✓  VALIDATION RÉUSSIE' if success else '✗  VALIDATION ÉCHOUÉE'}")
    print(f"{'='*65}\n")

    return success


def _print_metric(name, value, threshold):
    seuil = f"(seuil={threshold})" if threshold else ""
    verdict = ""
    if threshold is not None:
        verdict = "  ✓" if value >= threshold else "  ✗"
    print(f"  {name:<18}: {value:.3f}  {seuil}{verdict}")


def _save_annotated(img, detects, gt, path):
    """Sauvegarde l'image avec cercles prédits (vert) et GT (bleu)."""
    out = img.copy()

    # ── Ground truth (bleu)
    for cx, cy, r, lbl in gt:
        cv2.circle(out, (int(cx), int(cy)), int(r), (255, 100, 0), 2)
        draw_label(
            out,
            normalize_label(lbl),
            (int(cx - r), int(cy - r) - 12),
            color=(255, 100, 0)
        )

    # ── Prédictions
    for cx, cy, r, lbl, d_mm, conf in detects:
        color = (
            (0, 220, 80) if conf > 0.7 else
            (0, 165, 255) if conf > 0.4 else
            (0, 80, 255)
        )

        cv2.circle(out, (int(cx), int(cy)), int(r), color, 2)

        draw_label(
            out,
            f"{normalize_label(lbl)} {d_mm:.1f}mm {conf*100:.0f}%",
            (int(cx - r), int(cy + r) + 14),
            color=color
        )

    cv2.imwrite(path, out)


if __name__ == "__main__":
    evaluate_dataset(
        val_dir="data/validation",
        ann_dir="annotation",
        save_dir="output_validation",
        iou_threshold=0.30,
    )