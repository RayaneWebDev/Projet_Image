from utils import apply_hough_bias_correction


def estimate_diameter_mm(features, scale_factor, is_reference=False):
    """
    Estime le diamètre réel d'une pièce en millimètres.

    Applique la correction du biais HoughCircles avant la conversion,
    sauf pour la pièce de référence.

    Paramètres :
        features      : dict avec "diameter_pixels"
        scale_factor  : mm/pixel (depuis utils.compute_scale_factor)
        is_reference  : True si c'est la pièce de référence (pas de correction)

    Retourne :
        float : diamètre estimé en mm
    """
    diameter_px = features["diameter_pixels"]

    if not is_reference:
        diameter_px = apply_hough_bias_correction(diameter_px)

    return diameter_px * scale_factor