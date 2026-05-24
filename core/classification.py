"""
Classification des pièces euro par k-NN pondéré (méthode classique).

Stratégie en deux étapes :
  1. Scale Factor global (calibration pixel→mm contraint par couleur)
  2. k-NN sur ring_features pour gold et silver si la confiance est suffisante

La détection bimétal distingue 1€ et 2€ lorsque le scale factor
est ambigu (seulement 2,5 mm d'écart).

Auteurs : Équipe ImageGroupe
Date    : 2026
"""
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
    """
    Charge la base k-NN depuis le fichier .npy et calcule les statistiques
    de normalisation (moyenne et écart-type par feature).

    La normalisation z-score est précalculée ici une seule fois au chargement
    pour ne pas la recalculer à chaque prédiction.

    Returns:
        (X, y, mu, sigma) ou (None, None, None, None) si fichier absent/corrompu
    """
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
    """
    Prédit le label d'une pièce par k-NN pondéré par distance inverse.

    Seuls les exemples dont le label correspond au groupe couleur détecté
    sont considérés. Les features sont normalisées en z-score avant le calcul
    de distance pour équilibrer l'influence de chaque dimension.

    Le vote final pondère chaque voisin par 1/distance : les voisins
    les plus proches ont donc plus d'influence sur le résultat.

    Args:
        ring_features: vecteur numpy de features (240 dims)
        color:         couleur détectée ('bronze', 'gold', 'silver', 'unknown')
        k:             nombre de voisins à considérer
    Returns:
        (label_prédit, confiance) ou (None, 0.0) si base indisponible
    """
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


def _detect_bimetal(crop, diff_sat_thr=18, diff_val_thr=10):
    """
    Détecte si une pièce est bimétal en comparant la saturation HSV
    du centre (20% du rayon) vs l'anneau intermédiaire (35-44% du rayon).

    Les pièces bimétal (1€ et 2€) ont un centre de couleur différente
    de leur anneau extérieur, créant une différence de saturation mesurable.

    Args:
        crop:         image BGR (crop circulaire de la pièce)
        diff_sat_thr: seuil minimum de différence de saturation (défaut 18)
        diff_val_thr: seuil minimum de différence de luminosité (défaut 10)
    Returns:
        (is_bimetal, sat_sign) où sat_sign > 0 indique un centre doré (→ 2€)
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

    # sat_sign > 0 : centre plus saturé que anneau → centre doré → 2€
    # sat_sign < 0 : centre moins saturé que anneau → centre argenté → 1€
    sat_sign = sat_c - sat_r
    return (diff_sat > diff_sat_thr) and (diff_val > diff_val_thr), sat_sign


def classify_all(features_list):
    """
    Classifie toutes les pièces d'une image en deux étapes :

    Étape 1 — Scale factor global contraint par couleur :
        Cherche sf tel que diameter_px * sf ≈ diameter_mm pour toutes les pièces.
        Minimise l'erreur totale en testant toutes les hypothèses de référence,
        contraintes par couleur (ex: une pièce bronze ne peut référencer que 1/2/5ct).

    Étape 2 — Correction par k-NN :
        Pour gold et silver, le k-NN est prioritaire sur le scale factor
        si sa confiance dépasse un seuil (0.60 pour gold, 0.55 pour silver).
        Pour bronze et unknown, le scale factor seul est utilisé.

    Cas spécial pièce unique :
        Le scale factor est trivial (erreur=0 pour tout candidat), donc inutilisable.
        On utilise directement le k-NN sans contrainte de couleur (tous candidats).

    Cas spécial 2€ :
        Le 2€ est souvent détecté gold à cause de son anneau doré.
        Si bimétal détecté sur une pièce gold -> reclassement silver.
        Séparation finale 1€/2€ par diamètre estimé + diff_sat.

    Args:
        features_list: liste de dicts retournée par extract_features()
    Returns:
        liste de (label, d_mm, confiance) dans le même ordre
    """
    if not features_list:
        return []

    # --- Cas spécial : pièce unique ---
    # Le scale factor est trivial (erreur = 0 pour tous les candidats),
    # donc on utilise directement le k-NN sans contrainte de couleur.
    if len(features_list) == 1:
        feat  = features_list[0]
        rf    = feat.get("ring_features")
        crop  = feat.get("crop_cv2")
        color = feat.get("color_label", "unknown")

        knn_label, knn_conf = _knn_predict(rf, "unknown", k=1)

        if knn_label is None:
            cands     = COLOR_CANDS.get(color, list(COIN_DIAMETERS_MM.keys()))
            knn_label = cands[0]
            knn_conf  = 0.0

        final_label = knn_label

        # Raffinement 1€/2€ par bimetal : sat_sign > 0 = centre doré = 2€
        if final_label in ["1 Euro", "2 Euro"] and crop is not None:
            is_bimetal, sat_sign = _detect_bimetal(crop)
            if is_bimetal:
                final_label = "2 Euro" if sat_sign > 0 else "1 Euro"

        d_mm = COIN_DIAMETERS_MM[final_label]
        conf = max(0.3, knn_conf)
        return [(final_label, d_mm, conf)]

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

        # 2€ souvent detecte gold a cause de son anneau dore -> reclasser silver
        if color == "gold" and crop is not None:
            is_bimetal, _ = _detect_bimetal(crop)
            if is_bimetal:
                color = "silver"
                cands = COLOR_CANDS["silver"]

        # Label par scale factor (nearest neighbor sur diametre)
        sf_label            = min(cands, key=lambda c: abs(COIN_DIAMETERS_MM[c] - d_mm))
        k_val               = 1 if color == "silver" else 5
        knn_label, knn_conf = _knn_predict(rf, color, k=k_val)

        # k-NN prioritaire si confiance suffisante, sinon scale factor
        # Pour silver (1€/2€) : k-NN toujours utilisé car scale factor peu fiable (2.5mm)
        if color == "gold" and knn_label is not None and knn_conf > 0.56:
            final_label = knn_label
        elif color == "bronze" and knn_label is not None and knn_conf > 1.0:
            final_label = knn_label
        elif color == "silver" and knn_label is not None:
            final_label = knn_label
        else:
            final_label = sf_label

        # Separation 1€/2€ par bimetal : sat_sign > 0 = centre doré = 2€
        if final_label in ["1 Euro", "2 Euro"] and crop is not None:
            is_bimetal, sat_sign = _detect_bimetal(crop)
            if is_bimetal:
                final_label = "2 Euro" if sat_sign > 0 else "1 Euro"

        # Confiance basée sur l'écart entre d_mm estimé et le diamètre réel du label
        dist = abs(COIN_DIAMETERS_MM[final_label] - d_mm)
        conf = max(0.3, 1.0 - dist / 5.0)
        results.append((final_label, d_mm, conf))

    return results


def classify_piece(features, all_features):
    """
    Wrapper pour compatibilité avec main.py.
    Appelle classify_all() et retourne le résultat de la pièce demandée.

    Args:
        features:     dict de features d'une pièce
        all_features: liste complète des features de l'image
    Returns:
        (label, d_mm, confiance)
    """
    all_results = classify_all(all_features)
    idx = all_features.index(features)
    if idx < len(all_results):
        return all_results[idx]
    return "unknown", features["diameter_pixels"], 0.0


def get_coin_value(label):
    """
    Retourne la valeur en euros d'un label de pièce.

    Args:
        label: str parmi '1 cent', '2 cent', ..., '2 Euro'
    Returns:
        float (ex: 0.50 pour '50 cent'), ou 0.0 si label inconnu
    """
    return COIN_VALUES_EUR.get(label, 0.0)