__author__ = "Polyarnyi Nickolay"

import cv2

import os
import sys
import getopt
import numpy as np

from leporis.utils.image_utils import resize_to, draw_points


def _to_absolute_points(points, w):
    return (np.array(points) * w).astype(np.int32)


def run_choose_pixel(image_file,
                     *, mouse_moved=None, mouse_left_click=None, mouse_right_click=None, width=1024):
    empty_handler = lambda pixels, pixel: None
    mouse_moved = mouse_moved or empty_handler
    mouse_left_click = mouse_left_click or empty_handler
    mouse_right_click = mouse_right_click or empty_handler

    src_img = cv2.imread(image_file)
    h, w = src_img.shape[:2]

    win_name = 'Leporis. Choose pixels'
    src_img = resize_to(src_img, width)
    cv2.imshow(win_name, src_img)

    points = []

    def on_mouse(event, x, y, flags, param):
        nonlocal points

        x, y = x / width, y / width
        point = np.array([x, y])

        event_handler = None
        update_image = False
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append(point)
            event_handler = mouse_left_click
            update_image = True
        elif event == cv2.EVENT_RBUTTONDOWN:
            points = points[:-1]
            event_handler = mouse_right_click
            update_image = True
        elif event == cv2.EVENT_MOUSEMOVE:
            event_handler = mouse_moved
        if update_image:
            cv2.imshow(win_name, draw_points(src_img.copy(), np.array(points), color=(0, 0, 255)))
        if event_handler is not None:
            event_handler(_to_absolute_points(points, w), _to_absolute_points(point, w))

    cv2.setMouseCallback(win_name, on_mouse)
    while True:
        if cv2.waitKey(20) & 0xFF == 27:
            break
    return _to_absolute_points(points, w)


def main(argv):
    usages = 'Usage: <image_file>\n\n' \
             'Left click - add pixel to points, and print points list.\n' \
             'Right click - delete last pixel from points.\n' \
             'Escape - to finish.'
    try:
        opts, args = getopt.getopt(argv, "h")
        if len(args) != 1:
            raise getopt.GetoptError('Image file argument missing!')
    except getopt.GetoptError:
        print(usages)
        return 2
    image_file = args[0]
    pixels = run_choose_pixel(image_file,
                              mouse_moved=lambda pixels, pixel: print(pixel),
                              mouse_left_click=lambda pixels, pixel: print(pixels),
                              mouse_right_click=lambda pixels, pixel: print(pixels))
    print(pixels)
    return 0


if __name__ == "__main__":
   sys.exit(main(sys.argv[1:]))
