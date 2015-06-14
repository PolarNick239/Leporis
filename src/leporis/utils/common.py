__author__ = "Polyarnyi Nickolay"

import numpy as np


def norm2(a):
    return (a * a).sum(-1)


def norm(a):
    return np.sqrt(norm2(a))


def homogenize(a, w=1.0):
    o = np.zeros((len(a), 1), a.dtype)
    o[:] = w
    return np.hstack([a, o])


def homo_translate(matrix, points):
    points = np.atleast_2d(points)
    if points.shape[-1] < matrix.shape[1]:
        points = homogenize(points)
    p = np.dot(points, matrix.T)
    return p[:, :-1] / p[:, -1, np.newaxis]


def iter_grouped(key, a, is_sorted=False):
    if not is_sorted:
        order = key.argsort()
        key, a = key[order], a[order]
    idx = [0] + list(np.diff(key).nonzero()[0] + 1) + [len(key)]
    for i1, i2 in zip(idx[:-1], idx[1:]):
        yield key[i1], a[i1:i2]
