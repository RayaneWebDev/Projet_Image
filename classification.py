def classify_piece(features):

    diameter = features["diameter_pixels"]

    # ⚠️ Valeurs exemple — à ajuster
    if diameter < 250:
        return "1 Euro"
    else:
        return "2 Euro"