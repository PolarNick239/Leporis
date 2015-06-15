__author__ = "Polyarnyi Nickolay"

import cv2

import os
import logging
import unittest
import numpy as np
from glob import glob

from leporis.algorithms import fast
from leporis.utils.image_utils import resize_to, to_gray_scale, draw_points
from leporis_test import test_commons

logger = logging.Logger(__name__)


class FastDetectorTest(unittest.TestCase):

    def setUp(self):
        self.img0 = cv2.imread(os.path.join(test_commons.resources_dir_path, 'fountain', '0000.jpg'))
        self.img0_512 = resize_to(self.img0, width=512)

        self.dummy_fast = fast.DummyFastDetector()

    def test_has_n_trues_in_a_row(self):
        self.assertFalse(fast.DummyFastDetector(3)._has_n_trues_in_a_row(np.array([False, True, False, True, True, False])))
        self.assertTrue(fast.DummyFastDetector(2)._has_n_trues_in_a_row(np.array([False, True, False, True, True, False])))
        self.assertFalse(fast.DummyFastDetector(4)._has_n_trues_in_a_row(np.array([False, True, False, True, True, True])))
        self.assertTrue(fast.DummyFastDetector(3)._has_n_trues_in_a_row(np.array([False, True, False, True, True, True])))

    def test1(self):
        points = self.dummy_fast.detect(to_gray_scale(self.img0_512))
        logger.warning('Points detected: {}.'.format(points.shape[0]))
        # cv2.imshow('Fast9', draw_points(self.img0_512, points))
        # cv2.waitKey()
        # TODO: assert something
