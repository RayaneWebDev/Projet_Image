"""
Segmentation des pièces euro dans une image.

Détecte les contours circulaires via la transformée de Hough et filtre
les doublons pour ne conserver qu'une détection par pièce physique.

Auteurs : Équipe ImageGroupe
Date    : 2026
"""
import cv2
import numpy as np


def segment_piece(image):
    """
    Détecte les pièces circulaires dans une image BGR.

    Applique un flou gaussien pour réduire le bruit, puis la transformée
    de Hough pour trouver les cercles. Les doublons (cercles dont les
    surfaces se chevauchent à plus de 70 %) sont éliminés en conservant
    le plus grand.

    Args:
        image (np.ndarray): Image BGR de dimensions (H, W, 3).

    Returns:
        list[tuple[int, int, int]]: Liste de triplets (x, y, rayon) en pixels,
            triés du plus grand au plus petit rayon.
    """
    h, w  = image.shape[:2]
    min_r = int(min(h, w) * 0.05)
    max_r = int(min(h, w) * 0.28)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (11, 11), 3)

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=int(min_r * 2),
        param1=100,
        param2=55,
        minRadius=min_r,
        maxRadius=max_r,
    )

    if circles is None:
        return []

    # Arrondi et cast en entiers pour des coordonnées pixel
    circles = np.round(circles[0]).astype(int)

    # Dédoublonnage : on parcourt du plus grand au plus petit rayon.
    # Un cercle est rejeté si son centre est à moins de 70 % de la somme
    # des rayons d'un cercle déjà retenu (chevauchement significatif).
    filtered = []
    for (x, y, r) in sorted(circles, key=lambda c: c[2], reverse=True):
        keep = True
        for (x2, y2, r2) in filtered:
            dist = np.sqrt((x - x2) ** 2 + (y - y2) ** 2)
            if dist < (r + r2) * 0.7:
                keep = False
                break
        if keep:
            filtered.append((x, y, r))

    return filtered
