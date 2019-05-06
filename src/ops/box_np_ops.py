from pathlib import Path
import numba
import numpy as np

def projection_matrix_to_CRT_kitti(proj):
    # P = C @ [R|T]
    # C is upper triangular matrix, so we need to inverse CR and use QR
    # stable for all kitti camera projection matrix
    CR = proj[0:3, 0:3]
    CT = proj[0:3, 3]
    RinvCinv = np.linalg.inv(CR)
    Rinv, Cinv = np.linalg.qr(RinvCinv)
    C = np.linalg.inv(Cinv)
    R = np.linalg.inv(Rinv)
    T = Cinv @ CT
    return C, R, T

def get_frustum(bbox_image, C, near_clip=0.001, far_clip=100):
    fku = C[0, 0]
    fkv = -C[1, 1]
    u0v0 = C[0:2, 2]
    z_points = np.array([near_clip] * 4 + [far_clip] * 4, dtype=C.dtype)[:, np.newaxis]
    b = bbox_image
    box_corners = np.array([[b[0], b[1]], [b[0], b[3]], [b[2], b[3]], [b[2], b[1]]],dtype=C.dtype)
    near_box_corners = (box_corners - u0v0) / np.array([fku / near_clip, -fkv / near_clip], dtype=C.dtype)
    far_box_corners = (box_corners - u0v0) / np.array([fku / far_clip, -fkv / far_clip], dtype=C.dtype)
    ret_xy = np.concatenate([near_box_corners, far_box_corners], axis=0)  # [8, 2]
    ret_xyz = np.concatenate([ret_xy, z_points], axis=1)
    return ret_xyz

def camera_to_lidar(points, r_rect, velo2cam):
    points_shape = list(points.shape[0:-1])
    if points.shape[-1] == 3:
        points = np.concatenate([points, np.ones(points_shape + [1])], axis=-1)
    lidar_points = points @ np.linalg.inv((r_rect @ velo2cam).T)
    return lidar_points[...,:3]


@numba.jit(nopython=True)
def corner_to_surfaces_3d_jit(corners):
    """convert 3d box corners from corner function above
    to surfaces that normal vectors all direct to internal.
    Args:
        corners (float array, [N, 8, 3]): 3d box corners. 
    Returns:
        surfaces (float array, [N, 6, 4, 3]): 
    """
    # box_corners: [N, 8, 3], must from corner functions in this module
    num_boxes = corners.shape[0]
    surfaces = np.zeros((num_boxes, 6, 4, 3), dtype=corners.dtype)
    corner_idxes = np.array([
        0, 1, 2, 3, 7, 6, 5, 4, 0, 3, 7, 4, 1, 5, 6, 2, 0, 4, 5, 1, 3, 2, 6, 7
    ]).reshape(6, 4)
    for i in range(num_boxes):
        for j in range(6):
            for k in range(4):
                surfaces[i, j, k] = corners[i, corner_idxes[j, k]]
    return surfacesef corner_to_surfaces_3d_jit()


def remove_outside_points(points, rect, Trv2c, P2, image_shape):
    C, R, T = projection_matrix_to_CRT_kitti(P2)
    image_bbox = [0, 0, image_shape[1], image_shape[0]]
    frustum = get_frustum(image_bbox, C)
    frustum -= T
    frustum = np.linalg.inv(R) @ frustum.T
    frustum = camera_to_lidar(frustum.T, rect, Trv2c)
    frustum_surfaces = corner_to_surfaces_3d_jit(frustum[np.newaxis, ...])
    indices = points_in_convex_polygon_3d_jit(points[:, :3], frustum_surfaces)
    points = points[indices.reshape([-1])]
    return points
