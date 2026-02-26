import cv2
import numpy as np


def extract_features(circles, image):
    """
    Extrait les features de chaque cercle détecté individuellement.

    CORRECTION MAJEURE : l'ancienne version dessinait tous les cercles
    sur un mask global, puis appelait findContours dessus. Quand des pièces
    étaient proches, leurs masques fusionnaient en un seul gros blob avec
    une mauvaise circularité → filtrés → seulement 2 pièces détectées sur 6.

    Nouvelle approche : chaque cercle Hough est traité séparément,
    sans passer par findContours. Les features sont calculées directement
    depuis les propriétés géométriques du cercle.

    Paramètres :
        circles : liste de tuples (cx, cy, r) retournée par segment_piece()
        image   : image OpenCV originale (numpy array BGR)

    Retourne :
        features_list    : liste de dicts avec les features de chaque pièce
        image_with_boxes : copie de l'image avec les bounding boxes dessinées
    """

    image_with_boxes = image.copy()
    features_list = []

    for (cx, cy, r) in circles:

        diameter_pixels = r * 2

        # Aire et périmètre théoriques du cercle (exact, pas besoin de contour)
        area = np.pi * r ** 2
        perimeter = 2 * np.pi * r
        circularity = 1.0  # Par définition, un cercle parfait a une circularité de 1

        # Bounding box
        x = max(cx - r, 0)
        y = max(cy - r, 0)
        x2 = min(cx + r, image.shape[1])
        y2 = min(cy + r, image.shape[0])
        w = x2 - x
        h = y2 - y

        # Dessin de la bounding box
        cv2.rectangle(
            image_with_boxes,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        features_list.append({
            "center": (cx, cy),
            "radius": r,
            "diameter_pixels": float(diameter_pixels),
            "area": area,
            "perimeter": perimeter,
            "circularity": circularity,
            "box": (x, y, w, h)
        })

    return features_list, image_with_boxes