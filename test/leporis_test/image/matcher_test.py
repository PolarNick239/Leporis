__author__ = "Polyarnyi Nickolay"


import cv2

import os
import unittest

from leporis.utils.image_utils import draw_key_points, resize_to, to_gray_scale
from leporis.image.matcher import FeatureMatcher, LKMatcher

from leporis_test import test_commons


class FeatureMatcherTest(unittest.TestCase):

    def setUp(self):
        self.img0 = cv2.imread(os.path.join(test_commons.resources_dir_path, 'IMG_0644.JPG'))
        self.img1 = cv2.imread(os.path.join(test_commons.resources_dir_path, 'IMG_0645.JPG'))
        self.gray_img0 = to_gray_scale(self.img0)
        self.gray_img1 = to_gray_scale(self.img1)

    def test(self):
        self.matcher = FeatureMatcher()
        self.lk_matcher = LKMatcher()

        kp0, descr0 = self.matcher.extract_features(self.img0)
        kp1, descr1 = self.matcher.extract_features(self.img1)

        match = self.matcher.match(kp0, descr0, kp1, descr1)

        # cv2.imshow('img0', draw_key_points(resize_to(self.img0), kp0))
        # cv2.imshow('img1', draw_key_points(resize_to(self.img1), kp1))
        # cv2.waitKey()
        # cv2.imshow('img0', draw_key_points(draw_key_points(resize_to(self.img0), kp0[match.f_idx[:, 0]], color=(0, 255, 0), thickness=2), kp0[match.h_idx[:, 0]], color=(255, 0, 0)))
        # cv2.imshow('img1', draw_key_points(draw_key_points(resize_to(self.img1), kp1[match.f_idx[:, 1]], color=(0, 255, 0), thickness=2), kp1[match.h_idx[:, 1]], color=(255, 0, 0)))
        # cv2.waitKey()

        kp0 = self.lk_matcher.extract_points(self.gray_img0)
        kp1 = self.lk_matcher.extract_points(self.gray_img1)

        match01, match10 = self.lk_matcher.match(self.gray_img0, kp0, self.gray_img1, kp1, match.h_mat)
        # cv2.imshow('img0', draw_key_points(resize_to(self.img0), kp0))
        # cv2.imshow('img1', draw_key_points(resize_to(self.img1), kp1))
        # cv2.waitKey()
        # cv2.imshow('img0', draw_key_points(draw_key_points(resize_to(self.img0), kp0[match01.p_idx0], color=(0, 255, 0), thickness=2), match10.points1, color=(255, 0, 0)))
        # cv2.imshow('img1', draw_key_points(resize_to(self.img1), match01.points1, color=(255, 0, 0)))
        # cv2.waitKey()
