from utils import COIN_DIAMETERS_MM, apply_hough_bias_correction


def classify_piece(features, scale_factor, is_reference=False):
    """
    Classifie une pièce euro par nearest-neighbor sur les diamètres officiels.

    Diamètres officiels utilisés (en mm) :
        1 cent   : 16.25 mm
        2 cent   : 18.75 mm
        10 cent  : 19.75 mm
        5 cent   : 21.25 mm
        20 cent  : 22.25 mm
        1 Euro   : 23.25 mm
        50 cent  : 24.25 mm
        2 Euro   : 25.75 mm

    Méthode :
        1. Corriger le biais de sous-estimation de HoughCircles
           (sauf pour la pièce de référence déjà correctement calibrée)
        2. Convertir en mm via scale_factor
        3. Nearest-neighbor : trouver le diamètre officiel le plus proche

    Paramètres :
        features      : dict avec "diameter_pixels"
        scale_factor  : mm/pixel (depuis utils.compute_scale_factor)
        is_reference  : True si c'est la pièce de référence (pas de correction de biais)

    Retourne :
        str : nom de la pièce identifiée (ex: "1 Euro", "20 cent", ...)
    """
    diameter_px = features["diameter_pixels"]

    # Correction du biais Hough (sauf pour la pièce de référence)
    if not is_reference:
        diameter_px_corr = apply_hough_bias_correction(diameter_px)
    else:
        diameter_px_corr = diameter_px

    # Conversion pixels → mm
    diameter_mm = diameter_px_corr * scale_factor

    # Nearest-neighbor : pièce dont le diamètre officiel est le plus proche
    best_name = min(COIN_DIAMETERS_MM.items(),
                    key=lambda x: abs(x[1] - diameter_mm))

    return best_name[0], diameter_mm