import cv2
import numpy as np
from segmentation import segment_piece
from features import extract_features
from classification import classify_piece
from regression import estimate_diameter_mm

def process_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("Erreur : image non trouvée")
        return

    # Redimensionnement pour affichage
    h, w = image.shape[:2]
    scale = 900 / max(h, w)
    image = cv2.resize(image, None, fx=scale, fy=scale)

    # Segmentation
    mask = segment_piece(image)

    # Extraction multi-pièces
    features_list, _ = extract_features(mask, image)

    if len(features_list) == 0:
        print("Aucune pièce détectée")
        return

    # Image finale
    segmented_image = image.copy()

    # Pour chaque pièce
    for features in features_list:
        x, y, w, h = features["box"]

        # Créer un masque pour cette pièce uniquement
        piece_mask = np.zeros(mask.shape, dtype=np.uint8)
        piece_mask[y:y+h, x:x+w] = mask[y:y+h, x:x+w]

        # Fusion vert transparent uniquement sur la pièce
        green_overlay = np.zeros_like(segmented_image)
        green_overlay[:] = (0, 255, 0)
        piece_mask_rgb = cv2.cvtColor(piece_mask, cv2.COLOR_GRAY2BGR)
        alpha = 0.4
        segmented_image = np.where(piece_mask_rgb==255,
                                   cv2.addWeighted(segmented_image, 1-alpha, green_overlay, alpha, 0),
                                   segmented_image)

        # Classification + régression
        classe = classify_piece(features)
        diameter_mm = estimate_diameter_mm(features)

        # Texte au-dessus de la pièce
        text = f"{classe} | {diameter_mm:.2f} mm"
        cv2.putText(segmented_image, text, (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Optionnel : contour autour de la pièce
        cv2.rectangle(segmented_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Affichage final
    cv2.imshow("Segmentation + Detection", segmented_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    process_image("img lounes medjbour.jpg")