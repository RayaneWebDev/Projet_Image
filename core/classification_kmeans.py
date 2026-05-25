"""
Classification des pièces euro par K-moyennes multi-centroïdes.

Pour chaque classe de chaque groupe chromatique, un K-moyennes est entraîné
séparément avec N_SUB sous-centroïdes par classe. Chaque centroïde est donc
garanti d'appartenir à une seule classe. La prédiction agrège les votes
pondérés par inverse de distance sur tous les centroïdes d'une même classe.

Pipeline identique à classification.py (scale factor + bimétal), avec
K-moyennes à la place du k-NN. Les features sont normalisées en z-score
avant l'entraînement et la prédiction.

Auteurs : Équipe ImageGroupe
Date    : 2026
"""
import cv2
import numpy as np
import os
from sklearn.cluster import KMeans
from core.utils import COIN_DIAMETERS_MM, COIN_VALUES_EUR

KNN_DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "model", "knn_database.npy"
)

COLOR_CANDS = {
    "bronze":  ["1 cent", "2 cent", "5 cent"],
    "gold":    ["10 cent", "20 cent", "50 cent"],
    "silver":  ["1 Euro", "2 Euro"],
    "unknown": list(COIN_DIAMETERS_MM.keys()),
}

# Nombre de sous-centroïdes par classe (même valeur pour tous les groupes).
# Plus N_SUB est grand, plus K-moyennes approche le comportement du k-NN
# (à la limite N_SUB = n_exemples ≈ 1-NN). L'optimum empirique est autour de 15-20.
N_SUB = 15

# Graines fixes pour reproductibilité
_SEEDS = {"bronze": 0, "gold": 100, "silver": 100, "unknown": 42}


def _load_kmeans():
    """
    Charge la base d'exemples et entraîne N_SUB centroïdes par classe.

    Pour chaque classe de chaque groupe couleur, un K-moyennes indépendant
    est entraîné sur les exemples de cette classe uniquement. Cela garantit
    que chaque centroïde appartient à une et une seule classe, contrairement
    à un K-moyennes global qui peut "voler" des exemples entre classes.

    Returns:
        tuple (models, mu, sigma) où models est un dict
        {couleur: {"centroids": ndarray (N*N_SUB, 240), "labels": list[str]}}
        ou (None, None, None) si fichier absent/corrompu.
    """
    if not os.path.exists(KNN_DB_PATH):
        return None, None, None
    try:
        data  = np.load(KNN_DB_PATH, allow_pickle=True).item()
        X     = data["X"].astype(np.float32)
        y     = data["y"]
        mu    = X.mean(axis=0)
        sigma = X.std(axis=0)
        sigma[sigma == 0] = 1.0
    except Exception:
        return None, None, None

    models = {}
    for color, cands in COLOR_CANDS.items():
        all_centroids = []
        all_labels    = []
        seed_offset   = _SEEDS.get(color, 42)
        for i, cls in enumerate(cands):
            mask_cls = (y == cls)
            if mask_cls.sum() == 0:
                continue

            X_cls  = X[mask_cls]
            X_norm = (X_cls - mu) / sigma
            k      = min(N_SUB, len(X_cls))

            km = KMeans(
                n_clusters   = k,
                init         = "k-means++",
                n_init       = 5,
                max_iter     = 300,
                random_state = seed_offset + i,
            )
            km.fit(X_norm)

            all_centroids.append(km.cluster_centers_)
            all_labels.extend([cls] * k)

        if not all_centroids:
            continue

        models[color] = {
            "centroids": np.vstack(all_centroids),
            "labels":    all_labels,
        }

    return (models if models else None), mu, sigma


_KM_MODELS, _KM_MU, _KM_SIGMA = _load_kmeans()


def _kmeans_predict(ring_features, color):
    """
    Prédit le label d'une pièce par vote agrégé sur les sous-centroïdes.

    Pour chaque classe du groupe, les poids inverses de distance de tous ses
    sous-centroïdes sont sommés. La classe avec le score agrégé le plus élevé
    est retenue. Cela est analogue au vote k-NN mais avec des représentants
    appris (centroïdes) plutôt que des exemples individuels.

    Args:
        ring_features (np.ndarray): vecteur 240 dims.
        color (str)               : groupe couleur ('bronze', 'gold', 'silver', 'unknown').

    Returns:
        tuple[str | None, float]: (label prédit, confiance) ou (None, 0.0).
    """
    if _KM_MODELS is None or ring_features is None:
        return None, 0.0

    color_key = color if color in _KM_MODELS else "unknown"
    if color_key not in _KM_MODELS:
        return None, 0.0

    model     = _KM_MODELS[color_key]
    centroids = model["centroids"]
    labels    = model["labels"]

    q_norm = (ring_features.astype(np.float32).flatten() - _KM_MU) / _KM_SIGMA
    dists  = np.sqrt(((centroids - q_norm) ** 2).sum(axis=1))

    # Agrégation des poids par classe : somme des 1/distance de tous ses centroïdes
    class_scores = {}
    for lbl, w in zip(labels, 1.0 / (dists + 1e-6)):
        class_scores[lbl] = class_scores.get(lbl, 0.0) + w

    best_label = max(class_scores, key=class_scores.get)
    total      = sum(class_scores.values())
    conf       = float(class_scores[best_label] / total) if total > 0 else 0.0

    return best_label, conf


def _detect_bimetal(crop, diff_sat_thr=18, diff_val_thr=10):
    """
    Détecte si une pièce est bimétal en comparant centre et anneau en HSV.

    Args:
        crop (np.ndarray) : image BGR du crop circulaire.
        diff_sat_thr (int): seuil différence de saturation.
        diff_val_thr (int): seuil différence de luminosité.

    Returns:
        tuple[bool, float]: (est_bimetal, sat_sign) — sat_sign > 0 → 2€, < 0 → 1€.
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
    sat_sign = sat_c - sat_r

    return (diff_sat > diff_sat_thr) and (diff_val > diff_val_thr), sat_sign


def classify_all_kmeans(features_list):
    """
    Classifie toutes les pièces d'une image avec K-moyennes.

    Pipeline identique à classify_all() (classification.py) :
      1. Scale factor global contraint par couleur.
      2. K-moyennes prioritaire sur SF pour gold (conf > 0.56) et silver (toujours).
      3. Bronze : scale factor toujours utilisé (seuil K-moyennes = 1.0, inatteignable).
      4. Bimétal : reclassement gold → silver, puis séparation 1€/2€.

    Args:
        features_list (list[dict]): sortie de extract_features().

    Returns:
        list[tuple[str, float, float]]: (label, d_mm, confiance) par pièce.
    """
    if not features_list:
        return []

    # --- Pièce unique ---
    if len(features_list) == 1:
        feat  = features_list[0]
        rf    = feat.get("ring_features")
        crop  = feat.get("crop_cv2")
        color = feat.get("color_label", "unknown")

        km_label, km_conf = _kmeans_predict(rf, "unknown")
        if km_label is None:
            cands    = COLOR_CANDS.get(color, list(COIN_DIAMETERS_MM.keys()))
            km_label = cands[0]
            km_conf  = 0.0

        final_label = km_label

        if final_label in ["1 Euro", "2 Euro"] and crop is not None:
            is_bimetal, sat_sign = _detect_bimetal(crop, diff_sat_thr=24)
            if is_bimetal:
                final_label = "2 Euro" if sat_sign > 0 else "1 Euro"

        d_mm = COIN_DIAMETERS_MM[final_label]
        conf = max(0.3, km_conf)
        return [(final_label, d_mm, conf)]

    diam_px    = [f["diameter_pixels"] for f in features_list]
    color_labs = [f.get("color_label", "unknown") for f in features_list]

    # --- Scale factor global contraint par couleur ---
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

        # Petites pièces bronze parfois détectées silver → reclasser unknown
        if color == "silver" and d_mm < 20.0:
            color = "unknown"
            cands = COLOR_CANDS["unknown"]

        # 2€ souvent détecté gold → reclasser silver si bimétal
        if color == "gold" and crop is not None:
            is_bimetal, _ = _detect_bimetal(crop)
            if is_bimetal:
                color = "silver"
                cands = COLOR_CANDS["silver"]

        sf_label          = min(cands, key=lambda c: abs(COIN_DIAMETERS_MM[c] - d_mm))
        km_label, km_conf = _kmeans_predict(rf, color)

        # Règles de priorité K-moyennes vs scale factor
        if color == "gold" and km_label is not None and km_conf > 0.56:
            final_label = km_label
        elif color == "bronze" and km_label is not None and km_conf > 1.0:
            final_label = km_label   # SF toujours
        elif color == "silver" and km_label is not None:
            final_label = km_label
        else:
            final_label = sf_label

        # Séparation 1€/2€ par bimétal
        if final_label in ["1 Euro", "2 Euro"] and crop is not None:
            is_bimetal, sat_sign = _detect_bimetal(crop, diff_sat_thr=24)
            if is_bimetal:
                final_label = "2 Euro" if sat_sign > 0 else "1 Euro"

        dist = abs(COIN_DIAMETERS_MM[final_label] - d_mm)
        conf = max(0.3, 1.0 - dist / 5.0)
        results.append((final_label, d_mm, conf))

    return results


def classify_piece_kmeans(features, all_features):
    """
    Wrapper de compatibilité — délègue à classify_all_kmeans().

    Args:
        features     (dict): features d'une pièce.
        all_features (list): liste complète des features de l'image.

    Returns:
        tuple[str, float, float]: (label, d_mm, confiance).
    """
    all_results = classify_all_kmeans(all_features)
    idx = all_features.index(features)
    if idx < len(all_results):
        return all_results[idx]
    return "unknown", features["diameter_pixels"], 0.0


def get_coin_value(label):
    """Retourne la valeur en euros d'un label de pièce."""
    return COIN_VALUES_EUR.get(label, 0.0)
