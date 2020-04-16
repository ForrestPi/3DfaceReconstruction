import numpy as np
import math
from math import cos, sin

def angle2matrix(angles):
    """
    Args:
        angles: [3, ]. x, y, z angles
        x: pitch
        y: yaw
        z: roll
    Returns:
        R: [3, 3]. rotation matrix
    """
    x, y, z = np.deg2rad(angles[0]), np.deg2rad(angles[1]), np.deg2rad(angles[2])
    Rx = np.array([[1,      0,       0], \
                   [0, cos(x), -sin(x)], \
                   [0, sin(x),  cos(x)]])

    Ry = np.array([[ cos(y), 0, sin(y)], \
                   [      0, 1,      0], \
                   [-sin(y), 0, cos(y)]])

    Rz = np.array([[cos(z), -sin(z), 0], \
                   [sin(z),  cos(z), 0], \
                   [     0,       0, 1]])

    R = Rz.dot(Ry.dot(Rx))
    return R.astype(np.float32)

def matrix2angle(R):
    """
    Args:
        R: [3, 3]. rotation matrix
    Returns:
        x: pitch
        y: yaw
        z: roll
    """
    assert(isRotationMatrix)
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0
    rx, ry, rz = np.rad2deg(x), np.rad2deg(y), np.rad2deg(z)
    return rx, ry, rz

def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

def similarity_transform(vertices, s, R, t3d):
    """
    Args:
        vertices: [n_vertices, 3]
        s: [1, ]. scale factor
        R: [3, 3]. rotation matrix
        t3d: [3, ]. 3d translation vector
    Returns:
        transformed vertices: [n_vertices, 3]
    """
    t3d = np.squeeze(np.array(t3d, dtype=np.float32))
    transformed_vertices = s * vertices.dot(R.T) + t3d[np.newaxis, :]
    return transformed_vertices

def to_image(vertices, h, w, is_perspective=False):
    """
    Args:
        vertices: [n_vertices, 3]
        h: height of the rendering
        w: width of the rendering
    Returns:
        image_vertices: [n_vertices, 3]
    """
    image_vertices = vertices.copy()
    if is_perspective:
        # if perspective, the projected vertices are normalized to [-1, 1]. so change it to image size first.
        image_vertices[:, 0] = image_vertices[:, 0] * w / 2
        image_vertices[:, 1] = image_vertices[:, 1] * h / 2
    image_vertices[:, 0] = image_vertices[:, 0] + w / 2
    image_vertices[:, 1] = image_vertices[:, 1] + h / 2
    # flip vertices along y-axis
    image_vertices[:, 1] = h - image_vertices[:, 1] - 1
    return image_vertices

def estimate_affine_matrix_3d_to_2d(X, x):
    """Using Golden Standard Algorithm for estimating an affine camera matrix P from world to image correspondences.
    Args:
        X: [n, 3]. 3d keypoints
        x: [n, 2]. corresponding 2d keypoints
    Returns:
        P_Affine: [3, 4]. Affine camera matrix
    """
    assert x.shape[0] == X.shape[0]
    assert x.shape[0] >= 4
    X = X.T  # (3, n)
    x = x.T  # (2, n)
    n = x.shape[1]

    ###---- 1. normalization
    ## 2d points
    mean = np.mean(x, 1)  # (2, )
    x = x - np.tile(mean[:, np.newaxis], [1, n])  # (2, n)
    average_norm = np.mean(np.sqrt(np.sum(x ** 2, 0)))
    scale = np.sqrt(2) / average_norm
    x = scale * x

    # T = [[scale,     0, -mean * scale], 
    #      [    0, scale, -mean * scale], 
    #      [    0,     0,             1 ]]
    T = np.zeros((3, 3), dtype=np.float32)
    T[0, 0] = T[1, 1] = scale
    T[:2, 2] = -mean * scale
    T[2, 2] = 1

    ## 3d points
    X_homo = np.vstack((X, np.ones((1, n))))  # (4, n)
    mean = np.mean(X, 1)  # (3, )
    X = X - np.tile(mean[:, np.newaxis], [1, n])  # (3, n)
    m = X_homo[: 3, :] - X
    average_norm = np.mean(np.sqrt(np.sum(X ** 2, 0)))
    scale = np.sqrt(3) / average_norm
    X = scale * X

    U = np.zeros((4, 4), dtype=np.float32)
    U[0, 0] = U[1, 1] = U[2, 2] = scale
    U[: 3, 3] = -mean * scale
    U[3, 3] = 1

    ###---- 2. equations
    A = np.zeros((n * 2, 8), dtype=np.float32)
    X_homo = np.vstack((X, np.ones((1, n)))).T
    A[: n, : 4] = X_homo
    A[n: , 4: ] = X_homo
    b = np.reshape(x, [-1, 1])  # (2n, 1)

    ###---- 3.solution
    p_8 = np.linalg.pinv(A).dot(b)  # (8, 2n) x (2n, 1) -> (8, 1)
    p = np.zeros((3, 4), dtype=np.float32)
    p[0, :] = p_8[:4, 0]
    p[1, :] = p_8[4:, 0]
    p[-1, -1] = 1

    ###---- 4. denormalization
    P_Affine = np.linalg.inv(T).dot(p.dot(U))
    return P_Affine

def P2sRt(P):
    """
    Args:
        P: [3, 4]. Affine Camera Matrix.
    Returns:
        s: [1, ]. scale factor.
        R: [3, 3]. rotation matrix
        t: [3, ]. translation
    """
    t = P[:, 3]
    R1 = P[0: 1, :3]
    R2 = P[1: 2, :3]
    s = (np.linalg.norm(R1) + np.linalg.norm(R2)) / 2.0
    r1 = R1 / np.linalg.norm(R1)
    r2 = R2 / np.linalg.norm(R2)
    r3 = np.cross(r1, r2)
    R = np.concatenate((r1, r2, r3), 0)
    return s, R, t

def get_colors_from_image(image, vertices):
    """
    Args:
        image: [h, w, 3].
        vertices: [n, 3]. 3->(x, y, z)
    Returns:
        colors: [n, 3]. 3->(b, g, r)
    """
    [h, w, _] = image.shape
    vertices[0, :] = np.minimum(np.maximum(vertices[0, :], 0), w - 1)  # x
    vertices[1, :] = np.minimum(np.maximum(vertices[1, :], 0), h - 1)  # y
    ind = np.round(vertices).astype(np.int32)
    colors = image[ind[1, :], ind[0, :], :]  # n x 3
    return colors
