import cv2
import numpy as np
from utils import COIN_DIAMETERS_MM, COIN_VALUES_EUR

SORTED_DIAMS = sorted(COIN_DIAMETERS_MM.items(), key=lambda x: x[1])


def _best_assignment(diameters_px, color_labels):
    """
    Trouve la meilleure assignation label→pièce par programmation dynamique.
    Cherche la combinaison qui minimise l'erreur globale sur les RATIOS
    entre pièces (invariant au scale_factor).
    """
    n = len(diameters_px)
    if n == 0:
        return [], 1.0

    # Trier par diamètre décroissant
    order   = sorted(range(n), key=lambda i: diameters_px[i], reverse=True)
    d_sort  = [diameters_px[i] for i in order]
    c_sort  = [color_labels[i] for i in order]

    # Candidats par couleur
    def candidates(color):
        if color == "bronze":
            return ["1 cent", "2 cent", "5 cent"]
        elif color == "gold":
            return ["10 cent", "20 cent", "50 cent", "1 Euro", "2 Euro"]
        elif color == "silver":
            return ["1 Euro", "2 Euro"]
        return list(COIN_DIAMETERS_MM.keys())

    # Scale factor optimal pour chaque hypothèse de référence
    best_labels = ["unknown"] * n
    best_sf     = 1.0
    best_error  = float('inf')

    # Tester toutes les hypothèses pour la pièce la plus grande
    for ref_label, ref_mm in COIN_DIAMETERS_MM.items():
        if ref_label not in candidates(c_sort[0]):
            continue
        sf = ref_mm / d_sort[0]

        # Assigner chaque pièce au label le plus proche en mm
        labels = []
        total_error = 0.0
        used = set()

        for i in range(n):
            d_mm   = d_sort[i] * sf
            cands  = [l for l in candidates(c_sort[i]) if l not in used]
            if not cands:
                cands = candidates(c_sort[i])

            best_l    = min(cands, key=lambda l: abs(COIN_DIAMETERS_MM[l] - d_mm))
            dist      = abs(COIN_DIAMETERS_MM[best_l] - d_mm)
            total_error += dist
            labels.append(best_l)
            used.add(best_l)

        if total_error < best_error:
            best_error  = total_error
            best_labels = labels
            best_sf     = sf

    # Remettre dans l'ordre original
    result = ["unknown"] * n
    for rank, orig_i in enumerate(order):
        result[orig_i] = best_labels[rank]

    return result, best_sf


def _detect_bimetal(crop):
    if crop is None or crop.size == 0:
        return False, 0.0
    h, w = crop.shape[:2]
    if h < 20 or w < 20:
        return False, 0.0

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    mask_c = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask_c, (w//2, h//2), int(w * 0.20), 255, -1)

    mask_r = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask_r, (w//2, h//2), int(w * 0.44), 255, -1)
    cv2.circle(mask_r, (w//2, h//2), int(w * 0.35),   0, -1)

    if np.count_nonzero(mask_c) < 10 or np.count_nonzero(mask_r) < 10:
        return False, 0.0

    sat_c = cv2.mean(hsv[:,:,1], mask=mask_c)[0]
    sat_r = cv2.mean(hsv[:,:,1], mask=mask_r)[0]
    val_c = cv2.mean(hsv[:,:,2], mask=mask_c)[0]
    val_r = cv2.mean(hsv[:,:,2], mask=mask_r)[0]

    diff_sat = abs(sat_c - sat_r)
    diff_val = abs(val_c - val_r)
    is_bimetal = (diff_sat > 25) and (diff_val > 20)

    return is_bimetal, diff_sat


def classify_all(features_list):
    if not features_list:
        return []

    COLOR_CANDS = {
        "bronze":  ["1 cent", "2 cent", "5 cent"],
        "gold":    ["10 cent", "20 cent", "50 cent"],
        "silver":  ["1 Euro", "2 Euro"],
        "unknown": list(COIN_DIAMETERS_MM.keys()),
    }

    diam_px    = [f["diameter_pixels"] for f in features_list]
    color_labs = [f.get("color_label", "unknown") for f in features_list]

    # ── Scale factor contraint par couleur ──────────────────
    best_sf  = None
    best_err = float("inf")

    for i, (d, color) in enumerate(zip(diam_px, color_labs)):
        for ref_label in COLOR_CANDS.get(color, list(COIN_DIAMETERS_MM.keys())):
            sf        = COIN_DIAMETERS_MM[ref_label] / d
            total_err = sum(
                min(abs(COIN_DIAMETERS_MM[l] - d2 * sf)
                    for l in COLOR_CANDS.get(c2, list(COIN_DIAMETERS_MM.keys())))
                for d2, c2 in zip(diam_px, color_labs)
            )
            if total_err < best_err:
                best_err = total_err
                best_sf  = sf

    sf = best_sf or 0.1

    # ── Classification ──────────────────────────────────────
    results = []
    for feat in features_list:
        d_mm   = feat["diameter_pixels"] * sf
        color  = feat.get("color_label", "unknown")
        crop   = feat.get("crop_cv2")
        cands  = COLOR_CANDS.get(color, list(COIN_DIAMETERS_MM.keys()))

        best_label = min(cands, key=lambda c: abs(COIN_DIAMETERS_MM[c] - d_mm))

        # Correction 1€ / 2€ par détection bimétal
        if best_label in ["1 Euro", "2 Euro"] and crop is not None:
            is_bimetal, diff = _detect_bimetal(crop)
            if is_bimetal:
                best_label = "2 Euro" if diff > 30 else "1 Euro"

        dist = abs(COIN_DIAMETERS_MM[best_label] - d_mm)
        conf = max(0.3, 1.0 - dist / 5.0)
        results.append((best_label, d_mm, conf))

    return results

def classify_piece(features, all_features):
    """
    Wrapper pour compatibilité avec main.py.
    Utilise classify_all en interne.
    """
    all_results = classify_all(all_features)
    idx = all_features.index(features)
    if idx < len(all_results):
        return all_results[idx]
    return "unknown", features["diameter_pixels"], 0.0


def get_coin_value(label):
    return COIN_VALUES_EUR.get(label, 0.0)