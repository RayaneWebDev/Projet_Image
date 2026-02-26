import cv2
import numpy as np

# Diamètres réels officiels des pièces euro (en mm)
COIN_DIAMETERS_MM = {
    "1 cent":  16.25,
    "2 cent":  18.75,
    "10 cent": 19.75,
    "5 cent":  21.25,
    "20 cent": 22.25,
    "1 Euro":  23.25,
    "50 cent": 24.25,
    "2 Euro":  25.75,
}

# Diamètre réel de la pièce de référence (1€ identifiée automatiquement)
REFERENCE_COIN_DIAMETER_MM = 23.25

# Seuil pour la correction du biais de HoughCircles
# HoughCircles sous-estime systématiquement les rayons des pièces plus petites
# Ce biais a été mesuré empiriquement à ~12px sur les pièces < 118px de diamètre
HOUGH_BIAS_CORRECTION_PX = 13
HOUGH_BIAS_THRESHOLD_PX = 118  # en dessous de ce seuil, on applique la correction


def find_reference_coin(features_list, gray_image):
    """
    Identifie la pièce de 1€ parmi les pièces détectées.

    Méthode : le 1€ est bimétallique (centre argenté sombre + anneau doré clair).
    On mesure la différence de luminosité entre le centre (r/3) et l'anneau extérieur.
    Si diff < -20 : centre plus sombre → pièce bimétallique → 1€ ou 2€.

    Retourne :
        index (int) de la pièce de référence, ou -1 si aucune trouvée.
    """
    best_idx = -1
    best_diff = 0

    for i, f in enumerate(features_list):
        cx, cy = f["center"]
        r = f["radius"]

        mask_inner = np.zeros(gray_image.shape, dtype=np.uint8)
        cv2.circle(mask_inner, (cx, cy), max(r // 3, 1), 255, -1)

        mask_outer = np.zeros(gray_image.shape, dtype=np.uint8)
        cv2.circle(mask_outer, (cx, cy), r - 3, 255, -1)
        cv2.circle(mask_outer, (cx, cy), int(r * 0.6), 0, -1)

        pixels_inner = gray_image[mask_inner == 255]
        pixels_outer = gray_image[mask_outer == 255]

        if len(pixels_inner) == 0 or len(pixels_outer) == 0:
            continue

        diff = float(np.mean(pixels_inner)) - float(np.mean(pixels_outer))

        # Cherche la pièce avec le diff le plus négatif (centre le plus sombre)
        if diff < best_diff:
            best_diff = diff
            best_idx = i

    # Seuil : diff < -20 indique un bimétallique fiable
    if best_diff < -20:
        return best_idx
    return -1


def compute_scale_factor(features_list, gray_image):
    """
    Calcule le facteur d'échelle pixels → mm.

    Identifie d'abord la pièce de 1€ (bimétallique) pour calibrer.
    Si aucune pièce bimétallique n'est trouvée, utilise la plus grande pièce
    comme référence (supposée = 1€).

    Retourne :
        scale_factor (float) : mm par pixel
        ref_idx (int)        : index de la pièce de référence dans features_list
    """
    ref_idx = find_reference_coin(features_list, gray_image)

    if ref_idx == -1:
        # Fallback : plus grande pièce = référence
        ref_idx = max(range(len(features_list)),
                      key=lambda i: features_list[i]["diameter_pixels"])

    ref_diam_px = features_list[ref_idx]["diameter_pixels"]
    scale_factor = REFERENCE_COIN_DIAMETER_MM / ref_diam_px
    return scale_factor, ref_idx


def apply_hough_bias_correction(diameter_px):
    """
    Corrige le biais de sous-estimation de HoughCircles.

    HoughCircles sous-estime systématiquement les rayons des petites pièces
    d'environ 12px. Cette correction s'applique uniquement aux pièces
    dont le diamètre détecté est inférieur à HOUGH_BIAS_THRESHOLD_PX.

    Paramètres :
        diameter_px : diamètre détecté par Hough (pixels)

    Retourne :
        diameter_px_corrigé (float)
    """
    if diameter_px < HOUGH_BIAS_THRESHOLD_PX:
        return diameter_px + HOUGH_BIAS_CORRECTION_PX
    return diameter_px


def calibrate_from_known_coin(diameter_pixels, coin_diameter_mm=23.25):
    """
    Calcule manuellement le scale factor depuis une pièce connue.

    Paramètres :
        diameter_pixels  : diamètre mesuré en pixels
        coin_diameter_mm : vrai diamètre (mm) de cette pièce

    Retourne :
        scale_factor (float)
    """
    if diameter_pixels <= 0:
        raise ValueError("Le diamètre en pixels doit être positif.")
    return coin_diameter_mm / diameter_pixels


def draw_label(image, text, position, font_scale=0.6, color=(0, 0, 255), thickness=2):
    """
    Dessine un label texte sur une image OpenCV en évitant les débordements.

    Paramètres :
        image      : image OpenCV (numpy array)
        text       : texte à afficher
        position   : tuple (x, y)
        font_scale : taille de la police
        color      : couleur BGR
        thickness  : épaisseur
    """
    x, y = position
    y = max(y, 15)  # ne pas déborder en haut de l'image
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, color, thickness)