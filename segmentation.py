import cv2
import numpy as np

def segment_piece(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Seuillage automatique Otsu
    _, thresh = cv2.threshold(
        blur, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Nettoyage morphologique
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Trouver les contours des pièces
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Créer une image de superposition verte transparente
    overlay = np.zeros_like(image, dtype=np.uint8)
    for contour in contours:
        cv2.drawContours(overlay, [contour], -1, (0, 255, 0), -1)  # Remplir en vert

    # Superposer sur l'image originale avec transparence
    result = cv2.addWeighted(image, 1.0, overlay, 0.5, 0)

    return result