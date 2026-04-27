import cv2
import numpy as np
import os
from core.utils import COIN_DIAMETERS_MM, COIN_VALUES_EUR

KNN_DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "model", "knn_database.npy"
)

# Candidats par groupe couleur : restreint la recherche k-NN et le scale factor
COLOR_CANDS = {
    "bronze":  ["1 cent", "2 cent", "5 cent"],
    "gold":    ["10 cent", "20 cent", "50 cent"],
    "silver":  ["1 Euro", "2 Euro"],
    "unknown": list(COIN_DIAMETERS_MM.keys()),
}


def _load_knn():
    """Charge la base k-NN et calcule les stats de normalisation."""
    if not os.path.exists(KNN_DB_PATH):
        return None, None, None, None
    try:
        data = np.load(KNN_DB_PATH, allow_pickle=True).item()
        X    = data["X"].astype(np.float32)
        y    = data["y"]
        # Moyenne et écart-type par feature sur la base d'entraînement
        mu   = X.mean(axis=0)
        sigma = X.std(axis=0)
        sigma[sigma == 0] = 1.0  # éviter division par zéro
        return X, y, mu, sigma
    except Exception:
        return None, None, None, None

_KNN_X, _KNN_Y, _KNN_MU, _KNN_SIGMA = _load_knn()


def _knn_predict(ring_features, color, k=5):
    """k-NN pondéré par distance, avec normalisation z-score des features."""
    if _KNN_X is None or ring_features is None:
        return None, 0.0

    cands = COLOR_CANDS.get(color, list(COIN_DIAMETERS_MM.keys()))
    mask  = np.array([lbl in cands for lbl in _KNN_Y])
    if mask.sum() == 0:
        return None, 0.0

    X_filt = _KNN_X[mask]
    y_filt = _KNN_Y[mask]

    # Normalisation z-score : centrer et réduire
    X_norm = (X_filt - _KNN_MU) / _KNN_SIGMA
    q_norm = (ring_features.astype(np.float32).flatten() - _KNN_MU) / _KNN_SIGMA

    dists = np.sqrt(((X_norm - q_norm) ** 2).sum(axis=1))

    k_eff     = min(k, len(dists))
    idx       = np.argsort(dists)[:k_eff]
    neighbors = y_filt[idx]

    votes = {}
    for lbl, d in zip(neighbors, dists[idx]):
        w = 1.0 / (d + 1e-6)
        votes[lbl] = votes.get(lbl, 0.0) + w

    best  = max(votes, key=votes.get)
    total = sum(votes.values())
    conf  = votes[best] / total if total > 0 else 0.0
    return best, conf


def _detect_bimetal(crop):
    """
    Detecte si une piece est bimetal en comparant la saturation HSV
    du centre (20% du rayon) vs l'anneau intermediaire (35-44% du rayon).
    Retourne (is_bimetal, diff_saturation).
    """
    if crop is None or crop.size == 0:
        return False, 0.0
    h, w = crop.shape[:2]
    if h < 20 or w < 20:
        return False, 0.0

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    ctr = (w // 2, h // 2)

    # Masque centre
    mask_c = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask_c, ctr, int(w * 0.20), 255, -1)

    # Masque anneau intermediaire
    mask_r = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask_r, ctr, int(w * 0.44), 255, -1)
    cv2.circle(mask_r, ctr, int(w * 0.35),   0, -1)

    if np.count_nonzero(mask_c) < 10 or np.count_nonzero(mask_r) < 10:
        return False, 0.0

    sat_c    = cv2.mean(hsv[:, :, 1], mask=mask_c)[0]
    sat_r    = cv2.mean(hsv[:, :, 1], mask=mask_r)[0]
    val_c    = cv2.mean(hsv[:, :, 2], mask=mask_c)[0]
    val_r    = cv2.mean(hsv[:, :, 2], mask=mask_r)[0]
    diff_sat = abs(sat_c - sat_r)
    diff_val = abs(val_c - val_r)

    # Seuils calibres empiriquement sur data/validation
    return (diff_sat > 15) and (diff_val > 10), diff_sat


def classify_all(features_list):
    """
    Classifie toutes les pieces d'une image en deux etapes :
      1. Scale factor global contraint par couleur -> label diametre
      2. k-NN sur ring_features -> correction si confiance suffisante
    Retourne une liste de (label, d_mm, confiance).
    """
    if not features_list:
        return []

    diam_px    = [f["diameter_pixels"] for f in features_list]
    color_labs = [f.get("color_label", "unknown") for f in features_list]

    # --- Scale factor global contraint par couleur ---
    # On cherche sf tel que : diametre_pixels * sf ≈ diametre_reel_mm
    # en minimisant l'erreur totale sur toutes les pieces de l'image.
    # La contrainte couleur evite que 10ct serve de reference pour des pieces bronze.
    best_sf, best_err = None, float("inf")
    for d, color in zip(diam_px, color_labs):
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

    results = []
    for feat in features_list:
        d_mm  = feat["diameter_pixels"] * sf
        color = feat.get("color_label", "unknown")
        crop  = feat.get("crop_cv2")
        rf    = feat.get("ring_features")
        cands = COLOR_CANDS.get(color, list(COIN_DIAMETERS_MM.keys()))

        # 2€ souvent detecte gold a cause de son anneau dore -> reclasser silver
        if color == "gold" and crop is not None:
            is_bimetal, _ = _detect_bimetal(crop)
            if is_bimetal:
                color = "silver"
                cands = COLOR_CANDS["silver"]

        # Label par scale factor (nearest neighbor sur diametre)
        sf_label            = min(cands, key=lambda c: abs(COIN_DIAMETERS_MM[c] - d_mm))
        knn_label, knn_conf = _knn_predict(rf, color, k=5)

        # k-NN prioritaire si confiance suffisante, sinon scale factor
        if color == "gold" and knn_label is not None and knn_conf > 0.60:
            final_label = knn_label
        elif color == "silver" and knn_label is not None and knn_conf > 0.55:
            final_label = knn_label
        else:
            final_label = sf_label

        # Separation 1€/2€ par bimetal + diametre
        # 2€ = 25.75mm, 1€ = 23.25mm -> seuil a 24.8mm
        if final_label in ["1 Euro", "2 Euro"] and crop is not None:
            is_bimetal, diff_sat = _detect_bimetal(crop)
            if is_bimetal:
                if d_mm > 24.8:
                    final_label = "2 Euro"
                elif d_mm < 22.0:
                    final_label = "1 Euro"
                else:
                    final_label = "2 Euro" if diff_sat > 20 else "1 Euro"

        dist = abs(COIN_DIAMETERS_MM[final_label] - d_mm)
        conf = max(0.3, 1.0 - dist / 5.0)
        results.append((final_label, d_mm, conf))

    return results


def classify_piece(features, all_features):
    """Wrapper pour compatibilite avec main.py — appelle classify_all en interne."""
    all_results = classify_all(all_features)
    idx = all_features.index(features)
    if idx < len(all_results):
        return all_results[idx]
    return "unknown", features["diameter_pixels"], 0.0


def get_coin_value(label):
    """Retourne la valeur en euros d'un label de piece."""
    return COIN_VALUES_EUR.get(label, 0.0)