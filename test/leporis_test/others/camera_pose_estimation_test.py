__author__ = "Polyarnyi Nickolay"


import cv2

import os
import unittest
import numpy as np
from glob import glob

from leporis.utils.image_utils import resize_to
from leporis_test import test_commons
from resources import points


class CameraPoseEstimationTest(unittest.TestCase):

    def test_on_two_descent_frames(self):
        points_3d = []
        key_to_index = {}

        for name, point in points.points.items():
            key_to_index[name] = len(points_3d)
            points_3d.append(point)
        points_3d = np.array(points_3d)

        frames_points_3d, frames_points_2d = [], []
        for frame_name, observations in points.observations.items():
            frame_points_2d, frame_points_3d = [], []
            for key, xy in observations.items():
                if key not in key_to_index:
                    continue
                xy = np.array(xy)
                frame_points_2d.append(xy)
                frame_points_3d.append(points_3d[key_to_index[key]])
            frame_points_2d, frame_points_3d = np.array(frame_points_2d, np.float32), np.array(frame_points_3d, np.float32)
            frames_points_3d.append(frame_points_3d)
            frames_points_2d.append(frame_points_2d)
        _, camera_matrix, distortion, r_vec, t_vec = cv2.calibrateCamera(frames_points_3d, frames_points_2d, (points.width, points.height), None, None)
        # TODO: assert smth

    def test_chessboard(self):
        chessboard_corners = np.zeros((8*8,3), np.float32)
        chessboard_corners[:, :2] = np.mgrid[0:8, 0:8].T.reshape(-1,2)

        frames_points_3d = []
        frames_points_2d = []

        for img_file in glob(os.path.join(test_commons.resources_dir_path, 'chessboard') + '/*.JPG'):
            img = resize_to(cv2.imread(img_file))
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            cv2.imshow('kb', gray)
            cv2.waitKey()

            found, corners = cv2.findChessboardCorners(gray, (8, 8), None)

            if found:
                frames_points_3d.append(chessboard_corners)

                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                frames_points_2d.append(corners)

                img = cv2.drawChessboardCorners(img, (7,6), corners, found)
                cv2.imshow('img',img)
                cv2.waitKey(500)

        cv2.destroyAllWindows()

        # _, camera_matrix, distortion, r_vec, t_vec = cv2.calibrateCamera(frames_points_3d, frames_points_2d, (points.width, points.height), None, None)
        # TODO: assert smth, add tests with classic chessboard datasets
