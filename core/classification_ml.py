"""
Classification des pieces euro par Random Forest.
Variante ML de classification.py — memes features, meme pipeline,
Random Forest a la place du k-NN.
"""
import cv2
import numpy as np
import os
from sklearn.ensemble import ExtraTreesClassifier
from core.utils import COIN_DIAMETERS_MM, COIN_VALUES_EUR


KNN_DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "model", "knn_database_rf.npy"
)

COLOR_CANDS = {
    "bronze":  ["1 cent", "2 cent", "5 cent"],
    "gold":    ["10 cent", "20 cent", "50 cent"],
    "silver":  ["1 Euro", "2 Euro"],
    "unknown": list(COIN_DIAMETERS_MM.keys()),
}


def _load_rf():
    """
    Charge la base d'exemples et entraine un Random Forest par groupe couleur.
    Le RF est robuste aux features non normalisees — pas de StandardScaler.
    Retourne un dict {couleur: rf} ou None si fichier absent.
    """
    if not os.path.exists(KNN_DB_PATH):
        return None
    try:
        data = np.load(KNN_DB_PATH, allow_pickle=True).item()
        X    = data["X"].astype(np.float32)
        y    = data["y"]
    except Exception:
        return None

    _SEEDS = {"bronze": 0, "gold": 100, "silver": 100, "unknown": 42}
    models = {}
    for color, cands in COLOR_CANDS.items():
        mask = np.array([lbl in cands for lbl in y])
        if mask.sum() < 2:
            continue
        X_filt = X[mask]
        y_filt = y[mask]

        clf = ExtraTreesClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=_SEEDS.get(color, 42),
        )
        clf.fit(X_filt, y_filt)
        models[color] = clf

    return models if models else None

_RF_MODELS = _load_rf()



def _rf_predict(ring_features, color, d_mm=-1.0, prior_sigma=0.0):
    """
    Predit le label d'une piece par Random Forest.
    d_mm       : diametre en mm estime par le scale factor (-1.0 = inconnu).
    prior_sigma: si > 0, multiplie les probas par un prior gaussien sur d_mm.
    Retourne (label, confiance) ou (None, 0.0) si modele indisponible.
    """
    if _RF_MODELS is None or ring_features is None:
        return None, 0.0

    color_key = color if color in _RF_MODELS else "unknown"
    if color_key not in _RF_MODELS:
        return None, 0.0

    rf = _RF_MODELS[color_key]
    q  = ring_features.astype(np.float32).flatten()

    proba = rf.predict_proba(q.reshape(1, -1))[0]

    if d_mm > 0 and prior_sigma > 0:
        prior = np.array([
            np.exp(-0.5 * ((COIN_DIAMETERS_MM.get(c, d_mm) - d_mm) / prior_sigma) ** 2)
            for c in rf.classes_
        ])
        proba = proba * prior
        proba = proba / proba.sum()

    label = rf.classes_[np.argmax(proba)]
    conf  = float(np.max(proba))
    return label, conf



def _detect_bimetal(crop, diff_sat_thr=18, diff_val_thr=10):
    """
    Detecte si une piece est bimetal en comparant la saturation HSV
    du centre (20% du rayon) vs l'anneau intermediaire (35-44% du rayon).
    diff_sat_thr: seuil saturation (18 defaut, 24+ pour split final 1€/2€).
    diff_val_thr: seuil luminosite (10 defaut, 20+ pour gold->silver strict).
    Retourne (is_bimetal, sat_sign).
    """
    if crop is None or crop.size == 0:
        return False, 0.0
    h, w = crop.shape[:2]
    if h < 20 or w < 20:
        return False, 0.0

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    ctr = (w // 2, h // 2)

    mask_c = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask_c, ctr, int(w * 0.20), 255, -1)

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
    sat_sign = sat_c - sat_r  # >0: centre doré→2€, <0: centre argenté→1€

    return (diff_sat > diff_sat_thr) and (diff_val > diff_val_thr), sat_sign


def classify_all_rf(features_list):
    """
    Classifie toutes les pieces d'une image avec Random Forest.
    Pipeline identique a classify_all() mais avec RF a la place du k-NN.
    Retourne une liste de (label, d_mm, confiance).
    """
    if not features_list:
        return []

    # Cas pièce unique : SF trivial, utiliser RF tous candidats
    if len(features_list) == 1:
        feat  = features_list[0]
        rf_f  = feat.get("ring_features")
        crop  = feat.get("crop_cv2")
        color = feat.get("color_label", "unknown")

        rf_label, rf_conf = _rf_predict(rf_f, "unknown")
        if rf_label is None:
            cands     = COLOR_CANDS.get(color, list(COIN_DIAMETERS_MM.keys()))
            rf_label  = cands[0]
            rf_conf   = 0.0

        final_label = rf_label

        if final_label in ["1 Euro", "2 Euro"] and crop is not None:
            is_bimetal, sat_sign = _detect_bimetal(crop, diff_sat_thr=24)
            if is_bimetal:
                final_label = "2 Euro" if sat_sign > 0 else "1 Euro"

        d_mm = COIN_DIAMETERS_MM[final_label]
        conf = max(0.3, rf_conf)
        return [(final_label, d_mm, conf)]

    diam_px    = [f["diameter_pixels"] for f in features_list]
    color_labs = [f.get("color_label", "unknown") for f in features_list]

    # Scale factor global contraint par couleur (erreur relative)
    best_sf, best_err = None, float("inf")
    for d, color in zip(diam_px, color_labs):
        for ref_label in COLOR_CANDS.get(color, list(COIN_DIAMETERS_MM.keys())):
            sf        = COIN_DIAMETERS_MM[ref_label] / d
            total_err = sum(
                min(abs(COIN_DIAMETERS_MM[l] - d2 * sf) / COIN_DIAMETERS_MM[l]
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

        # Petites pièces bronze parfois détectées silver (usure) → reclasser unknown
        if color == "silver" and d_mm < 20.0:
            color = "unknown"
            cands = COLOR_CANDS["unknown"]

        # Reclassement gold -> silver si bimetal detecte (cas 2€)
        if color == "gold" and crop is not None:
            is_bimetal, _ = _detect_bimetal(crop)
            if is_bimetal:
                color = "silver"
                cands = COLOR_CANDS["silver"]

        sf_label          = min(cands, key=lambda c: abs(COIN_DIAMETERS_MM[c] - d_mm))
        rf_label, rf_conf = _rf_predict(rf, color)

        # RF prioritaire si confiance suffisante, sinon scale factor
        # silver: SF fiable (ecart 2.5mm), bimetal gere la distinction 1€/2€
        if color == "gold" and rf_label is not None and rf_conf > 0.50:
            final_label = rf_label
        elif color == "bronze" and rf_label is not None and rf_conf > 0.60:
            final_label = rf_label
        else:
            final_label = sf_label

        # Separation 1€/2€ par bimetal : sat_sign > 0 = centre doré = 2€
        # Seuil 25 : evite les faux positifs (FP bimetal → erreurs 1€↔2€)
        if final_label in ["1 Euro", "2 Euro"] and crop is not None:
            is_bimetal, sat_sign = _detect_bimetal(crop, diff_sat_thr=24)
            if is_bimetal:
                final_label = "2 Euro" if sat_sign > 0 else "1 Euro"

        dist = abs(COIN_DIAMETERS_MM[final_label] - d_mm)
        conf = max(0.3, 1.0 - dist / 5.0)
        results.append((final_label, d_mm, conf))

    return results


def classify_piece_rf(features, all_features):
    """Wrapper pour compatibilite — appelle classify_all_rf en interne."""
    all_results = classify_all_rf(all_features)
    idx = all_features.index(features)
    if idx < len(all_results):
        return all_results[idx]
    return "unknown", features["diameter_pixels"], 0.0


def get_coin_value(label):
    """Retourne la valeur en euros d'un label de piece."""
    return COIN_VALUES_EUR.get(label, 0.0)