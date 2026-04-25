import numpy as np


def circle_iou(cx1, cy1, r1, cx2, cy2, r2):
    """
    Calcule l'IoU analytique exact entre deux cercles.
    Formule basee sur l'intersection de deux disques.
    """
    d = np.hypot(cx1 - cx2, cy1 - cy2)

    if d <= abs(r1 - r2):
        # Un cercle contient l'autre
        return (min(r1, r2) ** 2) / (max(r1, r2) ** 2)

    if d >= r1 + r2:
        # Aucune intersection
        return 0.0

    r1s   = r1 ** 2
    r2s   = r2 ** 2
    alpha = np.arccos(np.clip((d**2 + r1s - r2s) / (2 * d * r1), -1, 1))
    beta  = np.arccos(np.clip((d**2 + r2s - r1s) / (2 * d * r2), -1, 1))

    intersection = (r1s * (alpha - np.sin(2 * alpha) / 2) +
                    r2s * (beta  - np.sin(2 * beta)  / 2))
    union = np.pi * (r1s + r2s) - intersection
    return intersection / union if union > 0 else 0.0


def bbox_to_circle(pt1, pt2):
    """
    Convertit 2 points LabelMe (bounding box) en (cx, cy, r).
    pt1 = coin haut-gauche, pt2 = coin bas-droit.
    """
    cx = (pt1[0] + pt2[0]) / 2
    cy = (pt1[1] + pt2[1]) / 2
    r  = np.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1]) / 2
    return cx, cy, r


def normalize_label(label):
    """
    Normalise les variantes de labels vers le format canonique du pipeline.
    Ex: '2cents', '2cent', '2 cent' -> '2 cent'
    """
    s = label.strip().lower().replace(" ", "").replace("_", "")

    mapping = {
        "1cent":   "1 cent",  "2cent":   "2 cent",  "2cents":  "2 cent",
        "5cent":   "5 cent",  "5cents":  "5 cent",  "10cent":  "10 cent",
        "10cents": "10 cent", "20cent":  "20 cent", "20cents": "20 cent",
        "50cent":  "50 cent", "50cents": "50 cent", "1euro":   "1 Euro",
        "1euros":  "1 Euro",  "2euro":   "2 Euro",  "2euros":  "2 Euro",
    }
    return mapping.get(s, label)


def compute_metrics(detections, ground_truths, iou_threshold=0.5):
    """
    Compare les detections du pipeline aux annotations ground truth.

    detections    : liste de (cx, cy, r, label)  — sorties du pipeline
    ground_truths : liste de (cx, cy, r, label)  — depuis JSON LabelMe
    iou_threshold : seuil IoU pour valider une detection comme TP

    Retourne un dict avec TP, FP, FN, precision, rappel, F1,
    label_accuracy et le detail de chaque match (tp/fp/fn_details).
    """
    matched_gt       = set()
    tp_list, fp_list = [], []

    for (cx_d, cy_d, r_d, label_d) in detections:
        best_iou, best_j = 0.0, -1

        for j, (cx_g, cy_g, r_g, label_g) in enumerate(ground_truths):
            if j in matched_gt:
                continue
            iou = circle_iou(cx_d, cy_d, r_d, cx_g, cy_g, r_g)
            if iou > best_iou:
                best_iou, best_j = iou, j

        if best_iou >= iou_threshold and best_j >= 0:
            gt_label = normalize_label(ground_truths[best_j][3])
            pd_label = normalize_label(label_d)
            matched_gt.add(best_j)
            tp_list.append({
                "det_label":     pd_label,
                "gt_label":      gt_label,
                "iou":           best_iou,
                "label_correct": pd_label == gt_label,
            })
        else:
            fp_list.append({"det_label": normalize_label(label_d), "iou": best_iou})

    fn_list = [
        {"gt_label": normalize_label(ground_truths[j][3])}
        for j in range(len(ground_truths))
        if j not in matched_gt
    ]

    tp         = len(tp_list)
    fp         = len(fp_list)
    fn         = len(fn_list)
    tp_correct = sum(1 for t in tp_list if t["label_correct"])

    precision      = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall         = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1             = (2 * precision * recall / (precision + recall)
                      if (precision + recall) > 0 else 0.0)
    label_accuracy = tp_correct / tp if tp > 0 else 0.0

    return {
        "TP": tp, "FP": fp, "FN": fn,
        "precision":      precision,
        "recall":         recall,
        "f1":             f1,
        "label_accuracy": label_accuracy,
        "tp_details":     tp_list,
        "fp_details":     fp_list,
        "fn_details":     fn_list,
    }