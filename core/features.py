import cv2
import numpy as np

# Plages HSV pour classification couleur (H1, H2, S1, S2, V1, V2)
COLOR_RANGES = {
    "bronze": [
        (0,   15, 60, 255, 40, 210),
        (165, 180, 60, 255, 40, 210),
    ],
    "gold": [
        (15, 40, 40, 220, 60, 255),
    ],
    "silver": [
        (0, 180, 0, 45, 50, 255),
    ],
}


def _equalize_v(crop):
    """Egalisation d'histogramme globale sur le canal V (HSV)."""
    if crop is None or crop.size == 0:
        return crop
    hsv          = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def compute_ring_features(crop, n_bins=16):
    """
    Decoupe le crop en 3 anneaux concentriques (centre/milieu/bord)
    et calcule pour chaque anneau :
      - histogramme H, S, V (espace HSV)
      - histogramme magnitude gradient Sobel
      - histogramme direction gradient Sobel (invariant a la rotation)
    Retourne un vecteur numpy de taille 3 * 5 * n_bins = 240.
    """
    if crop is None or crop.size == 0:
        return np.zeros(3 * 5 * n_bins)

    # Egalisation histogramme sur V avant extraction des features
    crop = _equalize_v(crop)

    h, w   = crop.shape[:2]
    cx, cy = w // 2, h // 2
    r_max  = min(cx, cy)

    hsv  = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # Normalisation z-score sur le crop gris
    mu    = float(gray.mean())
    sigma = float(gray.std())
    if sigma > 0:
        gray = np.clip(
            (gray.astype(float) - mu) / sigma * 64 + 128, 0, 255
        ).astype(np.uint8)

    # Gradient Sobel : magnitude et direction
    gx    = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy    = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag   = cv2.magnitude(gx, gy)
    angle = cv2.phase(gx, gy, angleInDegrees=True)
    mag   = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    angle = (angle % 180).astype(np.uint8)

    rings = [(0.0, 0.3), (0.3, 0.6), (0.6, 1.0)]
    feature_vec = []

    for (r_min_ratio, r_max_ratio) in rings:
        r_min  = int(r_min_ratio * r_max)
        r_max_ = int(r_max_ratio * r_max)

        # Masque de l'anneau = disque externe - disque interne
        mask_outer = np.zeros((h, w), dtype=np.uint8)
        mask_inner = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask_outer, (cx, cy), r_max_, 255, -1)
        if r_min > 0:
            cv2.circle(mask_inner, (cx, cy), r_min, 255, -1)
        mask_ring  = cv2.subtract(mask_outer, mask_inner)

        # Exclure le fond noir du crop circulaire
        mask_valid = cv2.inRange(crop, (10, 10, 10), (255, 255, 255))
        mask_ring  = cv2.bitwise_and(mask_ring, mask_valid)

        if np.count_nonzero(mask_ring) == 0:
            feature_vec.extend([0.0] * (5 * n_bins))
            continue

        # Histogrammes H, S, V normalises
        for channel, (c_min, c_max) in enumerate([(0, 180), (0, 256), (0, 256)]):
            hist = cv2.calcHist([hsv], [channel], mask_ring, [n_bins], [c_min, c_max])
            hist = hist.flatten().astype(float)
            if hist.sum() > 0:
                hist /= hist.sum()
            feature_vec.extend(hist.tolist())

        # Histogramme magnitude gradient
        hist_g = cv2.calcHist([mag], [0], mask_ring, [n_bins], [0, 256])
        hist_g = hist_g.flatten().astype(float)
        if hist_g.sum() > 0:
            hist_g /= hist_g.sum()
        feature_vec.extend(hist_g.tolist())

        # Histogramme direction gradient
        hist_a = cv2.calcHist([angle], [0], mask_ring, [n_bins], [0, 180])
        hist_a = hist_a.flatten().astype(float)
        if hist_a.sum() > 0:
            hist_a /= hist_a.sum()
        feature_vec.extend(hist_a.tolist())

    return np.array(feature_vec)


def classify_color(crop):
    """
    Classe le crop en bronze / gold / silver par seuillage HSV.
    Retourne 'unknown' si aucune couleur n'atteint le seuil de 15%.
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
            m = cv2.inRange(hsv, (h1, s1, v1), (h2, s2, v2))
            combined = cv2.bitwise_or(combined, m)
        combined = cv2.bitwise_and(combined, mask_valid)
        scores[color_name] = np.count_nonzero(combined) / total

    best = max(scores, key=scores.get)
    return best if scores[best] > 0.15 else "unknown"


def extract_features(circles, image):
    """
    Pour chaque cercle detecte, extrait :
      - crop circulaire masque (fond noir)
      - couleur HSV (bronze / gold / silver)
      - diametre corrige via fitEllipse + biais HoughCircles x1.15
      - vecteur ring_features (240 dims) pour le k-NN
    Retourne (features_list, image_annotee).
    """
    image_out     = image.copy()
    features_list = []

    for (cx, cy, r) in circles:
        x  = max(cx - r, 0)
        y  = max(cy - r, 0)
        x2 = min(cx + r, image.shape[1])
        y2 = min(cy + r, image.shape[0])
        crop = image[y:y2, x:x2].copy()

        # Masque circulaire : met le fond en noir
        mask = np.zeros(crop.shape[:2], dtype=np.uint8)
        cv2.circle(mask, (r, r), r, 255, -1)
        crop_circle = cv2.bitwise_and(crop, crop, mask=mask)

        color_label   = classify_color(crop_circle)
        ring_features = compute_ring_features(crop_circle)

        # Correction perspective : fitEllipse + biais x1.15
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        diameter      = float(r * 2 * 1.15)
        ellipse_ratio = 1.0

        if contours:
            cnt = max(contours, key=cv2.contourArea)
            if len(cnt) >= 5:
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