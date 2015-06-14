__author__ = "Polyarnyi Nickolay"

import cv2


def to_gray_scale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def resize_to(img, width=512, height=None):
    h, w = img.shape[:2]
    if height is None:
        height = width * h // w
    return cv2.resize(img, (width, height))


def draw_key_points(img, key_points, radius=4, color=(255, 0, 0), thickness=1):
    h, w = img.shape[:2]
    for point in key_points:
        cv2.circle(img, (int(point[0]*w), int(point[1]*w)), radius, color, thickness)
    return img
