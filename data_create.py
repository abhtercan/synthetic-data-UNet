import os
import cv2
import random
import numpy as np

os.makedirs("data/images", exist_ok=True)
os.makedirs("data/masks", exist_ok=True)

def generate_surface(size=256):
    image = np.random.randint(120, 160, (size, size)).astype(np.uint8)
    mask = np.zeros((size, size), dtype=np.uint8)

    # Scratches
    for _ in range(random.randint(1, 4)):
        x1, y1 = random.randint(0, size), random.randint(0, size)
        x2, y2 = random.randint(0, size), random.randint(0, size)
        thickness = random.randint(1, 3)

        cv2.line(image, (x1, y1), (x2, y2), 220, thickness)
        cv2.line(mask, (x1, y1), (x2, y2), 1, thickness)

    # Stains
    for _ in range(random.randint(1, 3)):
        x, y = random.randint(0, size), random.randint(0, size)
        r = random.randint(10, 30)

        cv2.circle(image, (x, y), r, random.randint(60, 90), -1)
        cv2.circle(mask, (x, y), r, 1, -1)

    image = cv2.GaussianBlur(image, (5, 5), 0)
    noise = np.random.normal(0, 8, (size, size))
    image = np.clip(image + noise, 0, 255)

    return image.astype(np.uint8), mask
