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

    draw_label(image_out, f"TOTAL: {total:.2f} EUR", (10, image_out.shape[0] - 20))

    print("TOTAL =", total)

    cv2.imshow("Result", image_out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    process_image("8.jpg", debug=True)