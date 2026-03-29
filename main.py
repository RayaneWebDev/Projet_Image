import cv2
from segmentation import segment_piece
from features import extract_features
from classification import classify_piece, get_coin_value
from utils import draw_label


def process_image(image_path, debug=False):
    image = cv2.imread(image_path)

    if image is None:
        print("Erreur image")
        return

    # resize
    h, w = image.shape[:2]
    if max(h, w) > 1200:
        scale = 1200 / max(h, w)
        image = cv2.resize(image, None, fx=scale, fy=scale)


    circles = segment_piece(image)

    print(f"{len(circles)} pièces détectées")

    if len(circles) == 0:
        return

    features_list, image_out = extract_features(circles, image)

    total = 0

    for i, f in enumerate(features_list):

        label, d, conf = classify_piece(f, features_list)
        value = get_coin_value(label)
        total += value

        x, y, w, h = f["box"]
        cx, cy = f["center"]
        r = f["radius"]
        color_label = f["color_label"]

        # couleur affichage
        if conf > 0.7:
            draw_color = (0, 220, 80)
        elif conf > 0.4:
            draw_color = (0, 165, 255)
        else:
            draw_color = (0, 80, 255)

        overlay = image_out.copy()
        cv2.circle(overlay, (cx, cy), r, draw_color, -1)
        cv2.addWeighted(overlay, 0.2, image_out, 0.8, 0, image_out)

        text = f"{label} ({conf*100:.0f}%)"
        if debug:
            text += f" [{color_label}]"

        draw_label(image_out, text, (x, y - 10), draw_color)

        print(f"{label} | {conf*100:.0f}% | {value}€")

    print("TOTAL =", total)
    # Redimensionne en respectant le ratio
    max_w, max_h = 1400, 900
    h_out, w_out = image_out.shape[:2]
    scale = min(max_w / w_out, max_h / h_out, 1.0)
    display = cv2.resize(image_out, (int(w_out * scale), int(h_out * scale)),
                         interpolation=cv2.INTER_AREA)
    # Total écrit sur l'image redimensionnée
    draw_label(display, f"TOTAL: {total:.2f} EUR", (10, 30),
               color=(255, 220, 0), scale=0.8)
    cv2.imshow("Result", display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import os
    
    demo_dir = "demo_images_val"  # ← mets tes images ici
    
    if not os.path.isdir(demo_dir):
        print(f"Dossier '{demo_dir}' introuvable.")
    else:
        extensions = {".jpg", ".jpeg", ".png", ".bmp"}
        images = sorted([
            os.path.join(demo_dir, f)
            for f in os.listdir(demo_dir)
            if os.path.splitext(f)[1].lower() in extensions
        ])
        
        if not images:
            print(f"Aucune image dans '{demo_dir}'")
        else:
            print(f"{len(images)} image(s) trouvée(s) dans '{demo_dir}'")
            for img in images:
                print(f"\n--- {os.path.basename(img)} ---")
                process_image(img, debug=True)