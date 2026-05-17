import cv2
import numpy as np

# Plages HSV définissant chaque groupe couleur de pièces euro.
# Format par plage : (H_min, H_max, S_min, S_max, V_min, V_max)
# Les pièces bronze (1/2/5ct) ont une teinte rouge/orangée avec forte saturation.
# Les pièces gold (10/20/50ct) ont une teinte jaune/dorée.
# Les pièces silver (1€/2€) ont une très faible saturation (gris).
COLOR_RANGES = {
    "bronze": [
        (0,   15, 60, 255, 40, 210),
        (165, 180, 60, 255, 40, 210),  # zone rouge côté 180° de la roue HSV
    ],
    "gold":   [(15, 40, 40, 220, 60, 255)],
    "silver": [(0, 180,  0,  45, 50, 255)],
}


def _equalize_v(crop):
    """
    Applique une égalisation d'histogramme globale sur le canal V (luminosité)
    de l'espace HSV, sans toucher à la teinte ni à la saturation.

    Cela normalise l'éclairage du crop pour rendre les features
    plus robustes aux variations de luminosité entre images.

    Args:
        crop: image BGR (crop circulaire de la pièce)
    Returns:
        image BGR avec canal V égalisé
    """
    if crop is None or crop.size == 0:
        return crop
    hsv          = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def compute_ring_features(crop, n_bins=16):
    """
    Calcule un vecteur de features à partir du crop circulaire d'une pièce.

    Le crop est découpé en 3 anneaux concentriques :
      - centre  : 0 à 30% du rayon
      - milieu  : 30 à 60% du rayon
      - bord    : 60 à 100% du rayon

    Pour chaque anneau, on calcule 5 histogrammes normalisés :
      - H (teinte HSV)
      - S (saturation HSV)
      - V (luminosité HSV)
      - magnitude du gradient Sobel
      - direction du gradient Sobel (modulo 180°, invariant à la rotation)

    Avant l'extraction :
      - égalisation du canal V pour normaliser l'éclairage
      - filtre moyenneur 3×3 pour lisser les histogrammes
      - normalisation z-score du canal gris pour rendre les gradients
        indépendants de la luminosité absolue

    Args:
        crop:   image BGR carrée avec fond noir (crop circulaire)
        n_bins: nombre de bins par histogramme (défaut 16)
    Returns:
        vecteur numpy de taille 3 * 5 * n_bins = 240
    """
    if crop is None or crop.size == 0:
        return np.zeros(3 * 5 * n_bins)

    crop = _equalize_v(crop)
    crop = cv2.blur(crop, (3, 3))

    h, w   = crop.shape[:2]
    cx, cy = w // 2, h // 2
    r_max  = min(cx, cy)

    hsv  = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # Normalisation z-score : centre et réduit l'image en niveaux de gris
    # pour que les gradients Sobel soient indépendants de la luminosité
    mu, sigma = float(gray.mean()), float(gray.std())
    if sigma > 0:
        gray = np.clip(
            (gray.astype(float) - mu) / sigma * 64 + 128, 0, 255
        ).astype(np.uint8)

    # Gradient Sobel : magnitude (intensité du contour)
    #                  et direction (orientation, invariante par modulo 180°)
    gx    = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy    = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag   = cv2.normalize(cv2.magnitude(gx, gy), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    angle = (cv2.phase(gx, gy, angleInDegrees=True) % 180).astype(np.uint8)

    rings       = [(0.0, 0.3), (0.3, 0.6), (0.6, 1.0)]
    feature_vec = []

    for r_min_ratio, r_max_ratio in rings:
        r_min  = int(r_min_ratio * r_max)
        r_max_ = int(r_max_ratio * r_max)

        # Construction du masque de l'anneau par soustraction de deux disques
        mask_outer = np.zeros((h, w), dtype=np.uint8)
        mask_inner = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask_outer, (cx, cy), r_max_, 255, -1)
        if r_min > 0:
            cv2.circle(mask_inner, (cx, cy), r_min, 255, -1)
        mask_ring = cv2.subtract(mask_outer, mask_inner)

        # On exclut les pixels noirs (fond du crop circulaire)
        mask_valid = cv2.inRange(crop, (10, 10, 10), (255, 255, 255))
        mask_ring  = cv2.bitwise_and(mask_ring, mask_valid)

        if np.count_nonzero(mask_ring) == 0:
            feature_vec.extend([0.0] * (5 * n_bins))
            continue

        # Histogrammes H, S, V sur l'anneau
        for channel, (c_min, c_max) in enumerate([(0, 180), (0, 256), (0, 256)]):
            hist = cv2.calcHist([hsv], [channel], mask_ring, [n_bins], [c_min, c_max]).flatten().astype(float)
            if hist.sum() > 0:
                hist /= hist.sum()
            feature_vec.extend(hist.tolist())

        # Histogramme magnitude du gradient Sobel
        hist_g = cv2.calcHist([mag], [0], mask_ring, [n_bins], [0, 256]).flatten().astype(float)
        if hist_g.sum() > 0:
            hist_g /= hist_g.sum()
        feature_vec.extend(hist_g.tolist())

        # Histogramme direction du gradient Sobel
        hist_a = cv2.calcHist([angle], [0], mask_ring, [n_bins], [0, 180]).flatten().astype(float)
        if hist_a.sum() > 0:
            hist_a /= hist_a.sum()
        feature_vec.extend(hist_a.tolist())

    return np.array(feature_vec)


def classify_color(crop):
    """
    Détermine la couleur dominante d'un crop de pièce euro par seuillage HSV.

    Pour chaque groupe (bronze, gold, silver), on calcule le ratio de pixels
    correspondant aux plages HSV définies dans COLOR_RANGES.
    Le groupe avec le plus grand ratio est retenu s'il dépasse 15%.

    Args:
        crop: image BGR (crop circulaire de la pièce, fond noir)
    Returns:
        str parmi 'bronze', 'gold', 'silver', ou 'unknown'
    """
    if crop is None or crop.size == 0:
        return "unknown"

    hsv        = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask_valid = cv2.inRange(crop, (10, 10, 10), (255, 255, 255))
    total      = np.count_nonzero(mask_valid)
    if total == 0:
        return "unknown"

    scores = {}
    for color_name, ranges in COLOR_RANGES.items():
        combined = np.zeros(crop.shape[:2], dtype=np.uint8)
        for (h1, h2, s1, s2, v1, v2) in ranges:
            combined = cv2.bitwise_or(combined, cv2.inRange(hsv, (h1, s1, v1), (h2, s2, v2)))
        combined = cv2.bitwise_and(combined, mask_valid)
        scores[color_name] = np.count_nonzero(combined) / total

    best = max(scores, key=scores.get)
    return best if scores[best] > 0.15 else "unknown"


def extract_features(circles, image):
    """
    Extrait les features de chaque pièce détectée dans l'image.

    Pour chaque cercle HoughCircles :
      1. Découpe un crop carré autour du cercle
      2. Applique un masque circulaire (fond noir)
      3. Classifie la couleur (bronze / gold / silver)
      4. Calcule le vecteur ring_features pour le k-NN (240 dims)
      5. Corrige le diamètre via fitEllipse (perspective) × 1.15 (biais Hough)

    Le facteur ×1.15 compense la sous-estimation systématique du rayon
    par HoughCircles, calibré empiriquement sur data/validation.

    Args:
        circles: liste de (cx, cy, r) retournée par segment_piece()
        image:   image BGR originale
    Returns:
        (features_list, image_annotee) où features_list est une liste de dicts
    """
    image_out     = image.copy()
    features_list = []

    for (cx, cy, r) in circles:
        x  = max(cx - r, 0)
        y  = max(cy - r, 0)
        x2 = min(cx + r, image.shape[1])
        y2 = min(cy + r, image.shape[0])
        crop = image[y:y2, x:x2].copy()

        mask = np.zeros(crop.shape[:2], dtype=np.uint8)
        cv2.circle(mask, (r, r), r, 255, -1)
        crop_circle = cv2.bitwise_and(crop, crop, mask=mask)

        color_label   = classify_color(crop_circle)
        ring_features = compute_ring_features(crop_circle)

        # Estimation du diamètre par fitEllipse sur le contour du crop brut
        # (pas crop_circle, pour avoir un contour plus net sur le bord réel)
        # Correction perspective : morphologie + fitEllipse + biais x1.15
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Binarisation Otsu
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Element structurant elliptique adapte aux pieces circulaires
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # Fermeture : bouche les trous dans le masque de la piece
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Ouverture : supprime les artefacts parasites autour de la piece
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        diameter      = float(r * 2 * 1.15)  # fallback si pas d'ellipse
        ellipse_ratio = 1.0                   # rapport petit/grand axe (1.0 = cercle parfait)

        if contours:
            cnt = max(contours, key=cv2.contourArea)
            if len(cnt) >= 5:  # fitEllipse requiert au moins 5 points
                (_, _), (major, minor), _ = cv2.fitEllipse(cnt)
                diameter      = max(major, minor) * 1.15
                ellipse_ratio = min(major, minor) / max(major, minor)

        cv2.circle(image_out, (cx, cy), r, (0, 255, 80), 2)

        features_list.append({
            "center":          (cx, cy),
            "radius":          r,
            "diameter_pixels": float(diameter),
            "box":             (x, y, x2 - x, y2 - y),
            "crop_cv2":        crop_circle,
            "color_label":     color_label,
            "ellipse_ratio":   ellipse_ratio,
            "ring_features":   ring_features,
        })

    return features_list, image_out