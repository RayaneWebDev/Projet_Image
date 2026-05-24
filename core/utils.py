"""
Constantes et utilitaires partagés pour la reconnaissance de pièces euro.

Fournit les diamètres officiels, les valeurs faciales et une fonction
d'affichage de label utilisée par les scripts de démonstration et d'évaluation.

Auteurs : Équipe ImageGroupe
Date    : 2026
"""
import cv2

# Diamètres officiels des pièces euro en millimètres (source : BCE)
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

# Valeurs faciales en euros
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

# Mapping couleur → classes candidates (restreint SF et k-NN par groupe)
COLOR_TO_COINS = {
    "bronze":  ["1 cent", "2 cent", "5 cent"],
    "gold":    ["10 cent", "20 cent", "50 cent"],
    "silver":  ["1 Euro", "2 Euro"],
    "unknown": list(COIN_DIAMETERS_MM.keys()),
}


def draw_label(image, text, position, color=(0, 255, 100), scale=0.55):
    """
    Dessine un label texte avec fond noir sur une image BGR.

    Le fond noir garantit la lisibilité quelle que soit la couleur de l'image
    sous-jacente. La position est automatiquement corrigée pour rester dans
    les limites de l'image.

    Args:
        image    (np.ndarray)      : image BGR à annoter (modifiée en place).
        text     (str)             : texte à afficher.
        position (tuple[int, int]) : coin supérieur gauche (x, y) en pixels.
        color    (tuple[int, int, int]): couleur BGR du texte (défaut : vert clair).
        scale    (float)           : taille de la police (défaut : 0.55).
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, scale, 1)
    x, y = position
    # Correction pour ne pas déborder en haut ni à droite
    y = max(y, th + 6)
    x = min(x, image.shape[1] - tw - 4)
    cv2.rectangle(image, (x - 2, y - th - 4), (x + tw + 2, y + baseline), (0, 0, 0), -1)
    cv2.putText(image, text, (x, y), font, scale, color, 1, cv2.LINE_AA)
