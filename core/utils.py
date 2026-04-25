import cv2
import numpy as np

# Diamètres officiels des pièces euro en millimètres
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

# Mapping couleur → classes candidates (utilisé dans classification.py)
COLOR_TO_COINS = {
    "bronze":  ["1 cent", "2 cent", "5 cent"],
    "gold":    ["10 cent", "20 cent", "50 cent"],
    "silver":  ["1 Euro", "2 Euro"],
    "unknown": list(COIN_DIAMETERS_MM.keys()),
}

SORTED_DIAMETERS = sorted(COIN_DIAMETERS_MM.items(), key=lambda x: x[1])


def draw_label(image, text, position, color=(0, 255, 100), scale=0.55):
    """Dessine un label texte avec fond noir sur l'image."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, scale, 1)
    x, y = position
    y = max(y, th + 6)
    x = min(x, image.shape[1] - tw - 4)
    cv2.rectangle(image, (x - 2, y - th - 4), (x + tw + 2, y + baseline), (0, 0, 0), -1)
    cv2.putText(image, text, (x, y), font, scale, color, 1, cv2.LINE_AA)