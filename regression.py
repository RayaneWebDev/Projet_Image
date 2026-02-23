def estimate_diameter_mm(features, scale_factor=0.01):
    diameter_pixels = features["diameter_pixels"]
    return diameter_pixels * scale_factor
