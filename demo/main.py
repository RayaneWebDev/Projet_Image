"""
Demo visuelle du pipeline de reconnaissance de pieces euro.
Traite toutes les images du dossier demo_images_val/ et affiche
le resultat avec les labels et le total en euros.
"""
import os
import cv2

from core.segmentation import segment_piece
from core.features import extract_features
from core.classification import classify_piece, get_coin_value
from core.utils import draw_label


def process_image(image_path, debug=False):
    """
    Traite une image et affiche le resultat dans une fenetre OpenCV.
    debug=True : affiche aussi la couleur detectee (bronze/gold/silver).
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Erreur : impossible de lire {image_path}")
        return

    h, w = image.shape[:2]
    if max(h, w) > 1200:
        image = cv2.resize(image, None, fx=1200/max(h, w), fy=1200/max(h, w))

    circles = segment_piece(image)
    print(f"{len(circles)} piece(s) detectee(s)")

    if not circles:
        return

    features_list, image_out = extract_features(circles, image)
    total = 0.0

    for feat in features_list:
        label, d, conf = classify_piece(feat, features_list)
        value = get_coin_value(label)
        total += value

        cx, cy      = feat["center"]
        r           = feat["radius"]
        x, y, _, __ = feat["box"]
        color_label = feat["color_label"]

        # Couleur d'affichage selon la confiance
        if conf > 0.7:
            draw_color = (0, 220, 80)
        elif conf > 0.4:
            draw_color = (0, 165, 255)
        else:
            draw_color = (0, 80, 255)

        # Remplissage semi-transparent
        overlay = image_out.copy()
        cv2.circle(overlay, (cx, cy), r, draw_color, -1)
        cv2.addWeighted(overlay, 0.2, image_out, 0.8, 0, image_out)

        text = f"{label} ({conf*100:.0f}%)"
        if debug:
            text += f" [{color_label}]"
        draw_label(image_out, text, (x, y - 10), draw_color)

        print(f"  {label} | {conf*100:.0f}% | {value:.2f}EUR")

    print(f"  TOTAL = {total:.2f}EUR")

    # Affichage redimensionne
    max_w, max_h = 1400, 900
    h_out, w_out = image_out.shape[:2]
    scale   = min(max_w / w_out, max_h / h_out, 1.0)
    display = cv2.resize(image_out, (int(w_out * scale), int(h_out * scale)),
                         interpolation=cv2.INTER_AREA)
    draw_label(display, f"TOTAL: {total:.2f} EUR", (10, 30),
               color=(255, 220, 0), scale=0.8)
    cv2.imshow("Result", display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    demo_dir   = "demo/demo_images_val"
    extensions = {".jpg", ".jpeg", ".png", ".bmp"}

    if not os.path.isdir(demo_dir):
        print(f"Dossier '{demo_dir}' introuvable.")
    else:
        images = sorted([
            os.path.join(demo_dir, f)
            for f in os.listdir(demo_dir)
            if os.path.splitext(f)[1].lower() in extensions
        ])
        if not images:
            print(f"Aucune image dans '{demo_dir}'")
        else:
            print(f"{len(images)} image(s) dans '{demo_dir}'")
            for img_path in images:
                print(f"\n--- {os.path.basename(img_path)} ---")
                process_image(img_path, debug=True)