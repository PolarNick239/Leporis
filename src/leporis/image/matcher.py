#
# Copyright (c) 2015, Transas
# All rights reserved.
#

import cv2

import numpy as np
import logging

from leporis.utils import common
from leporis.utils import aabb


logger = logging.getLogger(__name__)

import sys
xfeatures2d = sys.modules['cv2.xfeatures2d']  # TODO: Fix OpenCV build for Python3.4 (may be fresh version already work)
SURF = xfeatures2d.SURF_create
SIFT = xfeatures2d.SIFT_create


class Match:

    serial_code = 1

    def __init__(self, fundamental_matrix, f_indices, homography_matrix, h_indices):
        self.f_mat = fundamental_matrix
        self.f_idx = f_indices
        self.h_mat = homography_matrix
        self.h_idx = h_indices

    def __getstate__(self):
        state = {'f_mat': self.f_mat,
                 'f_idx': self.f_idx,
                 'h_mat': self.h_mat,
                 'h_idx': self.h_idx}
        return state

    def __setstate__(self, state):
        self.f_mat = state['f_mat']
        self.f_idx = state['f_idx']
        self.h_mat = state['h_mat']
        self.h_idx = state['h_idx']


class LKMatch:

    serial_code = 1

    def __init__(self, points1, p_indices0, error01):
        self.points1 = points1
        self.p_idx0 = p_indices0
        self.error01 = error01

    def __getstate__(self):
        state = {'points1': self.points1,
                 'p_idx0': self.p_idx0,
                 'error01': self.error01}
        return state

    def __setstate__(self, state):
        self.points1 = state['points1']
        self.p_idx0 = state['p_idx0']
        self.error01 = state['error01']


def grid_suppress(points, weights, cell_size, cell_point_n):
    cells = np.array(points//cell_size, np.int32).view(np.int64).ravel()
    n = len(points)
    mask = np.zeros(n, np.bool8)
    for _, idxs in common.iter_grouped(cells, np.arange(n)):
        best_idxs = idxs[weights[idxs].argsort()[-cell_point_n:]]
        mask[best_idxs] = True
    return mask


def grid_suppress_kp(key_points, target_count, columns, img_size):
    if len(key_points) == 0:
        return key_points

    w, h = img_size
    rows = 1.0 * h / w * columns
    target_per_cell = int(target_count // (rows * columns))
    cell_size = w / columns

    key_points = np.array(key_points)
    cells = np.int32(np.int32([kp.pt for kp in key_points]) // cell_size)
    cells = cells.view(np.int64).ravel()
    res = []
    for _, kps in common.iter_grouped(cells, key_points):
        w = np.float32([kp.response for kp in kps])
        idx = w.argsort()[-target_per_cell:]
        res.extend(kps[idx])
    return res


class FeatureMatcher:

    def __init__(self, feature='surf', pixels_per_feature=1000,
                 min_match_n=15, grid_cells_cols=8, nn_ratio_thresh=0.75,
                 homography_threshold=0.1, fundamental_threshold=0.005):
        self.feature = feature
        self.pixels_per_feature = pixels_per_feature
        self.grid_cells_cols = grid_cells_cols
        self.min_match_n = min_match_n
        self.nn_ratio_thresh = nn_ratio_thresh
        self.homography_threshold = homography_threshold
        self.fundamental_threshold = fundamental_threshold

        if self.feature == 'surf':
            detector = SURF(hessianThreshold=200, nOctaves=2, nOctaveLayers=2, extended=False)
            self._create_detector = lambda feature_points: detector
            self._matcher = cv2.BFMatcher(cv2.NORM_L2)
        if self.feature == 'orb':
            self._create_detector = lambda feature_points: cv2.ORB(feature_points, nlevels=4)
            self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        if self.feature == 'sift':
            self._create_detector = lambda feature_points: SIFT(feature_points)
            self._matcher = cv2.BFMatcher(cv2.NORM_L2)
        if self.feature == 'akaze':
            detector = cv2.AKAZE()
            self._create_detector = lambda feature_points: detector
            self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    def extract_features(self, img):
        h, w = img.shape[:2]
        feature_target = w * h // self.pixels_per_feature
        detector = self._create_detector(feature_target * 3 // 2 if self.grid_cells_cols > 0 else feature_target)
        key_points = detector.detect(img)
        if self.grid_cells_cols > 0:
            key_points = grid_suppress_kp(key_points, feature_target, self.grid_cells_cols, (w, h))
        key_points, desc = detector.compute(img, key_points)

        if desc is None:
            desc = []
        
        key_points = np.float32([p.pt for p in key_points])
        key_points /= w
        return key_points, desc

    def _match_features(self, desc1, desc2):
        if len(desc1) == 0 or len(desc2) == 0:
            return np.int32([])
        matches = self._matcher.knnMatch(desc1, trainDescriptors=desc2, k=2)
        matches = [(m[0].queryIdx, m[0].trainIdx) for m in matches 
                   if len(m) == 2 and m[0].distance < m[1].distance * self.nn_ratio_thresh]
        return np.int32(matches)

    def _filter_matches_homography(self, p1, p2):
        H, mask = cv2.findHomography(p1, p2, cv2.RANSAC, self.homography_threshold)
        if mask is None:
            return np.eye(3), np.zeros(len(p1), np.bool8)
        return H, mask.ravel() != 0

    def _filter_matches_fundamental(self, p1, p2):
        F, mask = cv2.findFundamentalMat(p1, p2, cv2.FM_RANSAC, self.fundamental_threshold)
        return F, mask.ravel() != 0

    def match(self, kp1, desc1, kp2, desc2):
        f_data = None, []
        h_data = None, []

        m = self._match_features(desc1, desc2)
        if len(m) >= self.min_match_n:
            i, j = m.T
            p1, p2 = kp1[i], kp2[j]
            F, maskF = self._filter_matches_fundamental(p1, p2)
            inlierF = maskF.sum()
            if inlierF >= self.min_match_n:
                m = m[maskF]
                f_data = F, m

                i, j = m.T
                p1, p2 = kp1[i], kp2[j]
                H, maskH = self._filter_matches_homography(p1, p2)
                inlierH = maskH.sum()
                if inlierH >= self.min_match_n:
                    h_data = H, m[maskH]

        return Match(f_data[0], f_data[1], h_data[0], h_data[1])


def get_img_mask(img):
    if img.ndim == 2:
        return img != 0
    else:
        return (img != 0).any(-1)


def checked_flow(gray_img0, gray_img1, p0, max_err=1.0, win_size=15, max_level=3):
    lk_params = dict(winSize=(win_size, win_size),
                     maxLevel=max_level,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.1))
    h, w = gray_img0.shape[:2]
    p0 = p0 * w
    p_idx = np.arange(len(p0))

    img_size = np.int32([(0, 0), (w, h)])
    good = aabb.contains_point(img_size, p0)
    p0, p_idx = p0[good], p_idx[good]

    x, y = np.int32(p0.T)
    mask0, mask1 = get_img_mask(gray_img0), get_img_mask(gray_img1)
    good = mask0[y, x] & mask1[y, x]
    p0, p_idx = p0[good], p_idx[good]

    p1, _, _ = cv2.calcOpticalFlowPyrLK(gray_img0, gray_img1, p0, None, **lk_params)
    p0r, _, _ = cv2.calcOpticalFlowPyrLK(gray_img1, gray_img0, p1, None, **lk_params)
    err = common.norm(p0-p0r).ravel()
    good = err < max_err
    p1, p_idx, err = p1.reshape(-1, 2)[good], p_idx[good], err[good]
    return p1, p_idx, err


def adjust_h(homography_matrix, w0, w1):
    """
    :param homography_matrix: points with coordinates in [0..1] -> points with coordinates in [0..1]
    :return: homography_matrix: points with coordinates in [0...w0] -> points with coordinates in [0..w1]
    """
    return np.diag([w1, w1, 1.0]).dot(homography_matrix).dot(np.diag([1.0/w0, 1.0/w0, 1.0]))


class LKMatcher:

    def __init__(self, pixels_per_feature=500, min_match_n=20, grid_cells_cols=8):
        self.min_match_n = min_match_n
        self.pixels_per_feature = pixels_per_feature
        self.gftt = cv2.GFTTDetector_create(maxCorners=4096*4, qualityLevel=0.02, minDistance=9, blockSize=5)
        self.grid_cells_cols = grid_cells_cols

    def extract_points(self, gray_img):
        h, w = gray_img.shape[:2]
        kp = self.gftt.detect(gray_img)
        feature_num = h*w//self.pixels_per_feature
        kp = grid_suppress_kp(kp, feature_num, self.grid_cells_cols, (w, h))
        kp = np.float32([p.pt for p in kp])
        kp /= w
        return kp

    def match(self, gray_img0, kp0, gray_img1, kp1, guide_h01):
        (h0, w0), (h1, w1) = gray_img0.shape[:2], gray_img1.shape[:2]
        H01 = adjust_h(guide_h01, w0, w1)
        H10 = np.linalg.inv(H01)

        img1t = cv2.warpPerspective(gray_img1, H10, (w0, h0))
        p01, pidx01, err01 = checked_flow(gray_img0, img1t, kp0)
        p01 = common.homo_translate(H01, p01) / w1

        kp1t = np.float32(common.homo_translate(H10, kp1*w1)) / w1
        p10, pidx10, err10 = checked_flow(img1t, gray_img0, kp1t)
        p10 /= w0

        return LKMatch(p01, pidx01, err01) if len(p01) >= self.min_match_n else None,\
               LKMatch(p10, pidx10, err10) if len(p10) >= self.min_match_n else None
