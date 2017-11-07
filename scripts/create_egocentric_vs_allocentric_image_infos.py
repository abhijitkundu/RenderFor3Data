#!/usr/bin/python3

import numpy as np
import json
from collections import OrderedDict
from math import atan2, sqrt
from RenderFor3Data.image_dataset import ImageDataset, NoIndent, DatasetJSONEncoder


def wrap_to_pi(radians):
    """Wrap an angle (in radians) to [-pi, pi)"""
    # wrap to [0..2*pi]
    wrapped = radians % (2 * np.pi)
    # wrap to [-pi..pi]
    if wrapped >= np.pi:
        wrapped -= 2 * np.pi

    return wrapped


def is_rotation_matrix(R, atol=1e-6):
    """Checks if a matrix is a valid rotation matrix"""
    assert R.shape == (3, 3), "R is not a 3x3 matrix. R.shape = {}".format(R.shape)
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < atol


def skew_symm(x):
    """Returns skew-symmetric matrix from vector of length 3"""
    assert x.shape == (3,)
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]])


def rotation_from_two_vectors(a, b):
    """
    Returns rotation matrix that rotates a to be same as b
    See https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d/476311#476311
    """
    assert a.shape == (3,)
    assert b.shape == (3,)
    au = a / np.linalg.norm(a)
    bu = b / np.linalg.norm(b)
    ssv = skew_symm(np.cross(au, bu))
    c = au.dot(bu)
    assert c != -1.0, 'Cannot handle case when a and b are exactly opposite'
    R = np.eye(3) + ssv + np.matmul(ssv, ssv) / (1 + c)
    assert is_rotation_matrix(R)
    return R


def rotationX(angle):
    """Get the rotation matrix for rotation around X"""
    c = np.cos(angle)
    s = np.sin(angle)
    Rx = np.array([[1, 0, 0],
                   [0, c, -s],
                   [0, s, c]])
    return Rx


def rotationY(angle):
    """Get the rotation matrix for rotation around X"""
    c = np.cos(angle)
    s = np.sin(angle)
    Ry = np.array([[c, 0, s],
                   [0, 1, 0],
                   [-s, 0, c]])
    return Ry


def rotationZ(angle):
    """Get the rotation matrix for rotation around X"""
    c = np.cos(angle)
    s = np.sin(angle)
    Rz = np.array([[c, -s, 0],
                   [s, c, 0],
                   [0, 0, 1]])
    return Rz


def rotation_from_viewpoint(vp):
    """Get rotation matrix from viewpoint [azimuth, elevation, tilt]"""
    assert vp.shape == (3,)
    assert -np.pi <= vp[0] <= np.pi
    assert -np.pi / 2 <= vp[1] <= np.pi / 2
    assert -np.pi <= vp[2] <= np.pi

    R = rotationZ(-vp[2] - np.pi / 2).dot(rotationY(vp[1] + np.pi / 2)).dot(rotationZ(-vp[0]))
    assert is_rotation_matrix(R)
    return R


def viewpoint_from_rotation(R):
    """Returns viewpoint [azimuth, elevation, tilt] from rotation matrix"""
    assert is_rotation_matrix(R)

    plusMinus = -1
    minusPlus = 1

    I = 2
    J = 1
    K = 0

    res = np.empty(3)

    Rsum = sqrt((R[I, J] * R[I, J] + R[I, K] * R[I, K] + R[J, I] * R[J, I] + R[K, I] * R[K, I]) / 2)
    res[1] = atan2(Rsum, R[I, I])

    # There is a singularity when sin(beta) == 0
    if Rsum > 4 * np.finfo(float).eps:
        # sin(beta) != 0
        res[0] = atan2(R[J, I], minusPlus * R[K, I])
        res[2] = atan2(R[I, J], plusMinus * R[I, K])
    elif R[I, I] > 0:
        # sin(beta) == 0 and cos(beta) == 1
        spos = plusMinus * R[K, J] + minusPlus * R[J, K]  # 2*sin(alpha + gamma)
        cpos = R[J, J] + R[K, K]                         # 2*cos(alpha + gamma)
        res[0] = atan2(spos, cpos)
        res[2] = 0
    else:
        # sin(beta) == 0 and cos(beta) == -1
        sneg = plusMinus * R[K, J] + plusMinus * R[J, K]  # 2*sin(alpha - gamma)
        cneg = R[J, J] - R[K, K]                         # 2*cos(alpha - gamma)
        res[0] = atan2(sneg, cneg)
        res[2] = 0

    azimuth = -res[2]
    elevation = res[1] - np.pi / 2
    tilt = wrap_to_pi(-res[0] - np.pi / 2)

    return np.array([azimuth, elevation, tilt])


def main():
    """Main Function"""
    K = np.array([[721.5377, 0.0, 621.0], [0.0, 721.5377, 187.5], [0.0, 0.0, 1.0]])
    K_inv = np.linalg.inv(K)
    image_size = [1242, 375]
    vp = np.array([1.57079632679, 0.01265, 0.010868])
    dimension = np.array([3.691882, 1.936846, 1.587618])
    # shape_file = "CityShapes/Cars/Audi_A4/Audi_A4.obj"
    shape_file = "CityShapes/Cars/d2fb8f1c179dbf4e31269fbdcb80cff6/model.obj"
    image_center = np.array([145.0, 187.5])

    image_info = OrderedDict()
    image_info['image_intrinsic'] = NoIndent(K.tolist())
    image_info['image_size'] = NoIndent(image_size)
    image_info['image_file'] = "Allocentric.png"

    num_of_cars = 5
    step_x = (image_size[0] - 2 * image_center[0]) / (num_of_cars - 1)

    obj_infos = []
    for i in range(num_of_cars):
        obj_info = OrderedDict()
        obj_info['id'] = i + 1
        obj_info['shape_file'] = shape_file
        obj_info['dimension'] = NoIndent(dimension.tolist())
        obj_info['viewpoint'] = NoIndent(vp.tolist())
        center_proj = image_center + np.array([i * step_x, 0])
        obj_info['center_proj'] = NoIndent(center_proj.tolist())

        center_proj_ray = K_inv.dot(np.append(center_proj, 1))
        center_proj_ray = center_proj_ray / center_proj_ray[2] * 13.5
        obj_info['center_dist'] = np.linalg.norm(center_proj_ray)
        obj_infos.append(obj_info)

    image_info['object_infos'] = obj_infos

    with open("Allocentric.json", 'w') as f:
        json.dump(image_info, f, indent=2, separators=(',', ':'), cls=DatasetJSONEncoder)

    image_info['image_file'] = "Egocentric.png"
    R = rotation_from_viewpoint(vp)

    for obj_info in image_info['object_infos']:
        center_proj_ray = K_inv.dot(np.append(obj_info['center_proj'].value, 1))
        R_cp = rotation_from_two_vectors(center_proj_ray, np.array([0., 0., 1.]))
        Rvp = R_cp.dot(R)
        vp_dash = viewpoint_from_rotation(Rvp)
        obj_info['viewpoint'] = NoIndent(vp_dash.tolist())

    with open("Egocentric.json", 'w') as f:
        json.dump(image_info, f, indent=2, separators=(',', ':'), cls=DatasetJSONEncoder)


if __name__ == '__main__':
    main()
