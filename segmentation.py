import cv2
import numpy as np


def segment_piece(image):
    """
    Détecte les pièces via la transformée de Hough.

    Retourne une liste de cercles détectés : [(cx, cy, r), ...]
    au lieu d'un mask global (qui causait des fusions de contours
    lorsque des pièces étaient proches, rendant la détection incorrecte).

    Paramètres HoughCircles ajustés :
      - param2=40  : seuil plus strict → élimine les faux positifs
      - maxRadius=150 : exclut les artefacts trop grands
    """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)

    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=80,
        param1=100,
        param2=40,       # était 30 → trop permissif → faux positifs et artefacts
        minRadius=30,
        maxRadius=150    # était 200 → artefacts englobant plusieurs pièces à la fois
    )

    if circles is None:
        return []

    circles_int = np.around(circles[0]).astype(int)
    return [(int(cx), int(cy), int(r)) for cx, cy, r in circles_int]