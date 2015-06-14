__author__ = "Polyarnyi Nickolay"

import numpy as np


def contains_point(aabb, p):
    return np.all(aabb[0] <= p, -1) & np.all(p < aabb[1], -1)
