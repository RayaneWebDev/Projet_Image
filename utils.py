import cv2
import numpy as np

    # ── Diamètres officiels des pièces euro (mm) ──────────────────────────────
COIN_DIAMETERS_MM = {
        "1 cent":  16.25,
        "2 cent":  18.75,
        "5 cent":  21.25,
        "10 cent": 19.75,
        "20 cent": 22.25,
        "50 cent": 24.25,
        "1 Euro":  23.25,
        "2 Euro":  25.75,
    }

COIN_VALUES_EUR = {
        "1 cent":  0.01,
        "2 cent":  0.02,
        "5 cent":  0.05,
        "10 cent": 0.10,
        "20 cent": 0.20,
        "50 cent": 0.50,
        "1 Euro":  1.00,
        "2 Euro":  2.00,
    }

    # ── Mapping couleur → pièces compatibles ──────────────────────────────────
    # Permet de filtrer les candidats avant le nearest-neighbor sur le diamètre.
COLOR_TO_COINS = {
        "bronze":  ["1 cent", "2 cent", "5 cent"],
        "gold":    ["10 cent", "20 cent", "50 cent"],
        "silver":  ["1 Euro", "2 Euro"],
        "unknown": list(COIN_DIAMETERS_MM.keys()),   # pas de filtre si couleur indéterminée
    }

SORTED_DIAMETERS = sorted(COIN_DIAMETERS_MM.items(), key=lambda x: x[1])


def apply_hough_bias_correction(d, factor=1.6):
        """
        HoughCircles sous-estime légèrement le rayon (~4 %).
        Facteur calibré empiriquement ; ajustez sur votre base de validation.
        """
        return d * factor


def compute_scale_factor(features_list):
    if not features_list:
        return 0.1, 0

    diameters_px = sorted(
        [f["diameter_pixels"] for f in features_list],
        reverse=True
    )

    # Tester chaque hypothèse : quelle pièce est la plus grande ?
    # Garder le scale qui minimise l'erreur totale sur toutes les pièces
    best_sf    = None
    best_error = float('inf')

    for ref_name, ref_mm in COIN_DIAMETERS_MM.items():
        sf = ref_mm / diameters_px[0]
        total_error = sum(
            min(abs(d * sf - real_mm)
                for real_mm in COIN_DIAMETERS_MM.values())
            for d in diameters_px
        )
        if total_error < best_error:
            best_error = best_sf and best_error
            best_error = total_error
            best_sf    = sf

    largest = max(features_list, key=lambda f: f["diameter_pixels"])
    ref_idx = features_list.index(largest)
    return best_sf, ref_idx


def draw_label(image, text, position, color=(0, 255, 100), scale=0.55):
        """Dessine un label avec fond noir semi-opaque."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), baseline = cv2.getTextSize(text, font, scale, 1)
        x, y = position
        y = max(y, th + 6)
        x = min(x, image.shape[1] - tw - 4)
        cv2.rectangle(image, (x - 2, y - th - 4), (x + tw + 2, y + baseline), (0, 0, 0), -1)
        cv2.putText(image, text, (x, y), font, scale, color, 1, cv2.LINE_AA)


def preprocess_image(image):
        gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray)