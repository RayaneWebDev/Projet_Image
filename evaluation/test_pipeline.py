"""
Evalue le pipeline sur le jeu de test (data/test/).
Affiche les predictions par image et le resume global.
"""
import os
import cv2

from evaluation.evaluate import run_pipeline, load_annotation
from evaluation.metrics import compute_metrics, normalize_label
from core.utils import draw_label, COIN_VALUES_EUR

IOU_THRESHOLD = 0.5


def test_dataset(test_dir, ann_dir, save_dir="output_test"):
    os.makedirs(save_dir, exist_ok=True)

    image_ext = {".jpg", ".jpeg", ".png", ".bmp"}
    files = sorted([
        f for f in os.listdir(test_dir)
        if os.path.splitext(f)[1].lower() in image_ext
    ])

    if not files:
        print(f"Aucune image dans {test_dir}")
        return

    all_tp = all_fp = all_fn  = 0
    total_pred_eur = total_gt_eur = 0.0
    errors = []

    print(f"\n{'='*65}")
    print(f"  TEST — {len(files)} images")
    print(f"{'='*65}")

    for fname in files:
        img_path = os.path.join(test_dir, fname)
        ann_path = os.path.join(ann_dir, os.path.splitext(fname)[0] + ".json")

        img = cv2.imread(img_path)
        if img is None:
            print(f"  [SKIP] {fname}")
            continue

        h, w  = img.shape[:2]
        scale = min(1.0, 1200 / max(h, w))
        if scale < 1.0:
            img = cv2.resize(img, None, fx=scale, fy=scale)

        detects    = run_pipeline(img)
        pred_total = sum(
            COIN_VALUES_EUR.get(normalize_label(lbl), 0)
            for _, _, _, lbl, _, _ in detects
        )
        total_pred_eur += pred_total

        print(f"\n  {fname}")
        for cx, cy, r, lbl, d_mm, conf in detects:
            norm_lbl = normalize_label(lbl)
            val      = COIN_VALUES_EUR.get(norm_lbl, 0)
            print(f"    {norm_lbl:<10} {conf*100:.0f}%   {val:.2f}EUR")

        if os.path.exists(ann_path):
            gt       = load_annotation(ann_path, scale)
            gt_total = sum(COIN_VALUES_EUR.get(normalize_label(l), 0)
                           for _, _, _, l in gt)
            total_gt_eur += gt_total

            dets = [(cx, cy, r, lbl) for cx, cy, r, lbl, _, _ in detects]
            m    = compute_metrics(dets, gt, IOU_THRESHOLD)
            all_tp += m["TP"]
            all_fp += m["FP"]
            all_fn += m["FN"]

            err = abs(pred_total - gt_total)
            errors.append(err)
            print(f"    Predit {pred_total:.2f}EUR  |  Reel {gt_total:.2f}EUR"
                  f"  |  Erreur {err:.2f}EUR")
            print(f"    TP={m['TP']} FP={m['FP']} FN={m['FN']}"
                  f"  P={m['precision']:.2f} R={m['recall']:.2f}")
        else:
            print(f"    Predit {pred_total:.2f}EUR  (pas d'annotation)")

        _save_test_image(img, detects, os.path.join(save_dir, fname))

    # Resume global
    total_gt_annot = all_tp + all_fn
    g_prec = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0.0
    g_rec  = all_tp / total_gt_annot     if total_gt_annot > 0   else 0.0

    print(f"\n{'='*65}")
    print(f"  RESUME TEST")
    print(f"{'='*65}")
    print(f"  Pieces detectees : TP={all_tp} FP={all_fp} FN={all_fn}")
    print(f"  Precision        : {g_prec:.3f}")
    print(f"  Rappel           : {g_rec:.3f}")
    if errors:
        print(f"  Erreur monetaire moyenne : {sum(errors)/len(errors):.3f}EUR")
        print(f"  Erreur monetaire max     : {max(errors):.3f}EUR")
    print(f"  Total predit     : {total_pred_eur:.2f}EUR")
    if total_gt_eur > 0:
        print(f"  Total reel       : {total_gt_eur:.2f}EUR")
    print(f"  Images sauvegardees dans : {save_dir}/")
    print(f"{'='*65}\n")


def _save_test_image(img, detects, path):
    """Sauvegarde l'image annotee avec le total calcule."""
    out   = img.copy()
    total = 0.0
    for cx, cy, r, lbl, d_mm, conf in detects:
        norm_lbl = normalize_label(lbl)
        val      = COIN_VALUES_EUR.get(norm_lbl, 0)
        total   += val
        color    = (0, 220, 80) if conf > 0.7 else (0, 165, 255) if conf > 0.4 else (0, 80, 255)
        overlay  = out.copy()
        cv2.circle(overlay, (int(cx), int(cy)), int(r), color, -1)
        cv2.addWeighted(overlay, 0.18, out, 0.82, 0, out)
        cv2.circle(out, (int(cx), int(cy)), int(r), color, 2)
        draw_label(out, f"{norm_lbl} {conf*100:.0f}%  {val:.2f}EUR",
                   (int(cx - r), max(int(cy - r) - 12, 10)), color=color)
    draw_label(out, f"TOTAL: {total:.2f} EUR",
               (10, out.shape[0] - 20), color=(255, 220, 0), scale=0.7)
    cv2.imwrite(path, out)


if __name__ == "__main__":
    test_dataset(test_dir="data/test", ann_dir="annotations")