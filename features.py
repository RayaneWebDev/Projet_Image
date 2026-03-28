import cv2
import numpy as np

COLOR_RANGES = {
    "bronze": [
        # On accepte des couleurs plus sombres (V dès 30)
        # mais on exige une saturation très marquée (S > 80) pour ne pas confondre avec l'ombre portée sur la table
        (0, 15, 80, 255, 30, 200),
        (165, 180, 80, 255, 30, 200),
    ],
    "gold": [
        # L'or en plein soleil brille (V haut), mais dans l'ombre il devient "moutarde"
        # On élargit la plage de Valeur (V) de 40 à 255
        (15, 40, 60, 255, 40, 255),
    ],
    "silver": [
        # Le Silver est le plus dur à gérer avec l'ombre.
        # L'astuce : Le silver est TOUJOURS très peu saturé (S < 50).
        # Même dans l'ombre, une pièce d'argent reste "grise" (S bas).
        (0, 180, 0, 50, 40, 240),
    ],
}


def classify_color(crop):
    if crop is None or crop.size == 0:
        return "unknown"

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask_valid = cv2.inRange(crop, (10, 10, 10), (255, 255, 255))

    total_valid = np.count_nonzero(mask_valid)
    if total_valid == 0:
        return "unknown"

    scores = {}
    for color_name, ranges in COLOR_RANGES.items():
        combined_mask = np.zeros(crop.shape[:2], dtype=np.uint8)
        for (h1, h2, s1, s2, v1, v2) in ranges:
            m = cv2.inRange(hsv, (h1, s1, v1), (h2, s2, v2))
            combined_mask = cv2.bitwise_or(combined_mask, m)
        combined_mask = cv2.bitwise_and(combined_mask, mask_valid)
        scores[color_name] = np.count_nonzero(combined_mask) / total_valid

    best = max(scores, key=scores.get)
    return best if scores[best] > 0.15 else "unknown"


def extract_features(circles, image):
    image_out = image.copy()
    features_list = []

    for (cx, cy, r) in circles:

        x = max(cx - r, 0)
        y = max(cy - r, 0)
        x2 = min(cx + r, image.shape[1])
        y2 = min(cy + r, image.shape[0])

        crop = image[y:y2, x:x2].copy()

        # masque circulaire
        mask = np.zeros(crop.shape[:2], dtype=np.uint8)
        cv2.circle(mask, (r, r), r, 255, -1)
        crop_circle = cv2.bitwise_and(crop, crop, mask=mask)

        color_label = classify_color(crop_circle)

        # ── 🔥 CORRECTION PERSPECTIVE PAR ELLIPSE ───────────────
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        diameter = float(r * 2)  # fallback
        ellipse_ratio = 1.0

        if contours:
            cnt = max(contours, key=cv2.contourArea)

            if len(cnt) >= 5:
                ellipse = cv2.fitEllipse(cnt)
                (_, _), (major, minor), _ = ellipse

                # 🔥 on prend le GRAND axe = diamètre réel corrigé
                diameter = max(major, minor)

                ellipse_ratio = min(major, minor) / max(major, minor)

        # ── Dessin ──────────────────────────────────────────────
        cv2.circle(image_out, (cx, cy), r, (0, 255, 80), 2)

        features_list.append({
            "center": (cx, cy),
            "radius": r,
            "diameter_pixels": float(diameter),  # 🔥 CORRIGÉ
            "box": (x, y, x2 - x, y2 - y),
            "crop_cv2": crop_circle,
            "color_label": color_label,
            "ellipse_ratio": ellipse_ratio
        })

    return features_list, image_out