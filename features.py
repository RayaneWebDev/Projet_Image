import cv2
import numpy as np

def extract_features(mask, original_image):

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    image_with_boxes = original_image.copy()
    features_list = []

    for contour in contours:

        area = cv2.contourArea(contour)

        # Ignorer petits bruits
        if area < 500:
            continue

        perimeter = cv2.arcLength(contour, True)
        circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter != 0 else 0

        (xc, yc), radius = cv2.minEnclosingCircle(contour)
        diameter_pixels = radius * 2

        x, y, w, h = cv2.boundingRect(contour)

        # Dessiner rectangle
        cv2.rectangle(
            image_with_boxes,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        features_list.append({
            "area": area,
            "perimeter": perimeter,
            "circularity": circularity,
            "diameter_pixels": diameter_pixels,
            "box": (x, y, w, h)
        })

    return features_list, image_with_boxes