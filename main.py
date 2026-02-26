import cv2
import numpy as np
from segmentation import segment_piece
from features import extract_features
from classification import classify_piece
from regression import estimate_diameter_mm
from utils import compute_scale_factor, draw_label


def process_image(image_path):
    image = cv2.imread(image_path)

    if image is None:
        print(f"Erreur : image non trouvée → {image_path}")
        return

    # Redimensionner si trop grande
    h, w = image.shape[:2]
    max_size = 900
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        image = cv2.resize(image, None, fx=scale, fy=scale)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # --- Étape 1 : Détection des cercles ---
    circles = segment_piece(image)

    if len(circles) == 0:
        print("Aucune pièce détectée. Vérifiez les paramètres de HoughCircles.")
        return

    # --- Étape 2 : Extraction des features ---
    features_list, _ = extract_features(circles, image)

    print(f"{len(features_list)} pièce(s) détectée(s).")

    # --- Étape 3 : Calibration automatique ---
    # Identifie la pièce bimétallique (1€) pour calculer le scale factor
    scale_factor, ref_idx = compute_scale_factor(features_list, gray)
    print(f"Pièce de référence : pièce #{ref_idx + 1} "
          f"({features_list[ref_idx]['diameter_pixels']:.0f}px) "
          f"→ scale={scale_factor:.4f} mm/px")
    print()

    # --- Étape 4 : Rendu final ---
    segmented_image = image.copy()
    total_value = 0.0

    # Correspondance nom → valeur en euros
    coin_values = {
        "1 cent": 0.01, "2 cent": 0.02, "5 cent": 0.05,
        "10 cent": 0.10, "20 cent": 0.20, "50 cent": 0.50,
        "1 Euro": 1.00, "2 Euro": 2.00
    }

    for i, features in enumerate(features_list):

        cx, cy = features["center"]
        r = features["radius"]
        x, y, bw, bh = features["box"]
        is_ref = (i == ref_idx)

        # Overlay vert semi-transparent sur le disque
        overlay = segmented_image.copy()
        cv2.circle(overlay, (cx, cy), r, (0, 255, 0), -1)
        cv2.addWeighted(overlay, 0.30, segmented_image, 0.70, 0, segmented_image)

        # Cercle de détection vert
        cv2.circle(segmented_image, (cx, cy), r, (0, 255, 0), 2)

        # Classification + estimation diamètre
        classe, diameter_mm = classify_piece(features, scale_factor, is_reference=is_ref)
        diameter_mm_reg = estimate_diameter_mm(features, scale_factor, is_reference=is_ref)

        total_value += coin_values.get(classe, 0)

        # Label : nom + diamètre estimé
        text = f"{classe} | {diameter_mm_reg:.1f}mm"
        draw_label(segmented_image, text, (x, y - 10))

        # Debug console
        raw_px = features['diameter_pixels']
        print(f"  Pièce {i+1:>2}: {classe:10s} | {diameter_mm_reg:.1f}mm "
              f"| {raw_px:.0f}px brut "
              f"{'[REF]' if is_ref else ''}")

    print()
    print(f"  Valeur totale estimée : {total_value:.2f} €")

    cv2.imshow("Detection de pieces", segmented_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    process_image("img lounes medjbour.jpg")