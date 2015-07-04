__author__ = "Polyarnyi Nickolay"

import numpy as np
import pyopencl as cl


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


def create_cl_contex(device_type=None):
    if device_type is None:
        context, device = create_cl_contex(cl.device_type.GPU)
        if context is not None:
            return context, device
        context, device = create_cl_contex(cl.device_type.CPU)
        if context is not None:
            return context, device
        raise Exception('No GPU or CPU OpenCL devices!')

    platforms = cl.get_platforms()
    if len(platforms) == 0:
        return None, None

    context, device = None, None
    for platform in platforms:
        devices = platform.get_devices(device_type)
        if len(devices) != 0:
            device = devices[0]
            context = cl.Context([device])
            break

    return context, device

