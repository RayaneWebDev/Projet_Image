import cv2
import numpy as np


def segment_piece(image):
    h, w  = image.shape[:2]
    min_r = int(min(h, w) * 0.05)
    max_r = int(min(h, w) * 0.28)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 2)

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=int(min_r * 4),
        param1=100,
        param2=70,
        minRadius=min_r,
        maxRadius=max_r
    )

    if circles is None:
        return []

    circles = np.round(circles[0]).astype(int)

    # Déduplication : si deux cercles se chevauchent, garder le plus grand
    filtered = []
    for (x, y, r) in sorted(circles, key=lambda c: c[2], reverse=True):
        keep = True
        for (x2, y2, r2) in filtered:
            dist = np.sqrt((x - x2)**2 + (y - y2)**2)
            if dist < (r + r2) * 0.7:
                keep = False
                break
        if keep:
            filtered.append((x, y, r))

    return filtered