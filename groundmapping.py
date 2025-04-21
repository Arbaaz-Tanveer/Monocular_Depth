import cv2
import numpy as np
import time

# --------------------------
# Object Parameters
# --------------------------
# Ball radius (in same units as world coordinates, e.g., meters)
ball_radius = 0.11  # 11 cm

# --------------------------
# Calibration Utilities
# --------------------------
def compute_calibration_params(h, w, balance=1, distortion_param=0.05, show=False):
    K = np.array([[w/2, 0, w/2],
                  [0, w/2, h/2],
                  [0, 0, 1]], dtype=np.float32)
    D = np.array([[distortion_param], [0.0], [0.0], [0.0]], dtype=np.float32)
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K, D, (w, h), np.eye(3), balance=balance
    )
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), new_K, (w, h), cv2.CV_16SC2
    )
    return map1, map2, K, D, new_K

# --------------------------
# Coordinate Estimator Class
# --------------------------
class CoordinateEstimator:
    def __init__(self, image_width, image_height, fov_horizontal, fov_vertical, camera_height, camera_tilt=0):
        self.image_width = image_width
        self.image_height = image_height
        self.fov_h = np.radians(fov_horizontal)
        self.fov_v = np.radians(fov_vertical)
        self.camera_height = camera_height
        self.camera_tilt = np.radians(camera_tilt)

        # focal lengths in pixels
        self.focal_length_x = (image_width / 2) / np.tan(self.fov_h / 2)
        self.focal_length_y = (image_height / 2) / np.tan(self.fov_v / 2)

        # image center
        self.cx = image_width / 2
        self.cy = image_height / 2

        # precompute world coordinates lookup
        self.world_coords = self._precompute_world_coords()

    def _precompute_world_coords(self):
        xs, ys = np.meshgrid(np.arange(self.image_width), np.arange(self.image_height))
        x_norm = (xs - self.cx) / self.focal_length_x
        y_norm = (ys - self.cy) / self.focal_length_y
        rays = np.stack((x_norm, y_norm, np.ones_like(x_norm)), axis=2)
        norms = np.linalg.norm(rays, axis=2, keepdims=True)
        rays_normalized = rays / norms

        ct, st = np.cos(self.camera_tilt), np.sin(self.camera_tilt)
        R = np.array([[1, 0, 0], [0, ct, st], [0, -st, ct]])

        rays_flat = rays_normalized.reshape(-1, 3).T
        rays_world = (R @ rays_flat).T.reshape(self.image_height, self.image_width, 3)

        # intersection with ground plane y=0
        with np.errstate(divide='ignore', invalid='ignore'):
            t = -self.camera_height / rays_world[..., 1]
            t = np.where(np.abs(rays_world[..., 1]) < 1e-6, np.nan, t)
            t = np.where(t >= 0, np.nan, t)

        world_x = t * rays_world[..., 0]
        world_z = t * rays_world[..., 2]
        return np.stack((world_x, world_z), axis=2)

# --------------------------
# Pixel Undistortion & Ground Estimation
# --------------------------
def undistort_pixel(pixel_coords, K, D, new_K, show=False):
    pts = np.array(pixel_coords, dtype=np.float32).reshape(-1, 1, 2)
    undistorted = cv2.fisheye.undistortPoints(pts, K, D, None, new_K).reshape(-1, 2)
    undistorted_pixels = [tuple(pt) for pt in undistorted]
    if show:
        for orig, und in zip(pixel_coords, undistorted_pixels):
            print(f"Distorted {orig} -> Undistorted {und}")
    return undistorted_pixels


def pixel_to_ground(pixel_coords, estimator, K, D, new_K, show=False):
    """
    Convert image pixel coordinates (e.g., bottom of a detected ball) to ground-plane coordinates,
    then adjust for the object's radius to estimate its center.
    """
    undistorted = undistort_pixel(pixel_coords, K, D, new_K, show=show)
    ground_coords = []
    for x, y in undistorted:
        ix, iy = int(round(x)), int(round(y))
        if 0 <= ix < estimator.image_width and 0 <= iy < estimator.image_height:
            # raw world intersection (object front point)
            wx, wz = -estimator.world_coords[iy, ix]
            # compute direction along ground plane
            dist = np.hypot(wx, wz)
            if not np.isnan(dist) and dist > 1e-6:
                dx = wx / dist
                dz = wz / dist
                # move from front edge to object center
                cx = wx + dx * ball_radius
                cz = wz + dz * ball_radius
            else:
                cx, cz = wx, wz
            ground_coords.append((cx, cz))
        else:
            ground_coords.append((None, None))
    if show:
        for orig, und, center in zip(pixel_coords, undistorted, ground_coords):
            print(f"{orig} -> {und} -> Center: {center}")
    return ground_coords
