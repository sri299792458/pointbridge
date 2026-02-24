from dataclasses import dataclass
from typing import Any, Iterable, Optional

import numpy as np
import open3d as o3d
import pyrealsense2 as rs

def _get_xyz(depth_m: np.ndarray, K: np.ndarray) -> np.ndarray:
    h, w = depth_m.shape
    vu = np.mgrid[:h, :w]
    ones = np.ones((1, h, w), dtype=depth_m.dtype)
    uv1 = np.concatenate([vu[[1]], vu[[0]], ones], axis=0)
    uv1_prime = uv1 * depth_m
    return np.linalg.inv(K) @ uv1_prime.reshape(3, -1)


def _get_intrinsics(stream_profile: "rs.stream_profile"):
    prof = rs.video_stream_profile(stream_profile)
    return prof.get_intrinsics()


def _get_K(intrinsics) -> np.ndarray:
    K = np.eye(3)
    K[0, 0] = intrinsics.fx
    K[1, 1] = intrinsics.fy
    K[0, 2] = intrinsics.ppx
    K[1, 2] = intrinsics.ppy
    return K


@dataclass
class _CameraHandle:
    serial: str
    pipeline: "rs.pipeline"
    align: Optional["rs.align"]
    K: np.ndarray
    depth_scale: float
    T_world_camera: np.ndarray


class RealSensePerception:
    def __init__(
        self,
        camera_configs: Iterable,
        segmenter: Optional[Any] = None,
        voxel_size: Optional[float] = None,
    ):
        self._segmenter = segmenter
        self._voxel_size = voxel_size
        self._cameras = []
        self._last_debug_frames = []

        for cam in camera_configs:
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_device(cam.serial)
            config.enable_stream(rs.stream.depth, cam.width, cam.height, rs.format.z16, cam.fps)
            config.enable_stream(rs.stream.color, cam.width, cam.height, rs.format.rgb8, cam.fps)
            profile = pipeline.start(config)

            depth_sensor = profile.get_device().first_depth_sensor()
            depth_scale = depth_sensor.get_depth_scale()
            if cam.align_to_color:
                align = rs.align(rs.stream.color)
                color_profile = profile.get_stream(rs.stream.color)
                intr = _get_intrinsics(color_profile)
            else:
                align = None
                depth_profile = profile.get_stream(rs.stream.depth)
                intr = _get_intrinsics(depth_profile)
            K = _get_K(intr)

            self._cameras.append(
                _CameraHandle(
                    serial=cam.serial,
                    pipeline=pipeline,
                    align=align,
                    K=K,
                    depth_scale=depth_scale,
                    T_world_camera=cam.T_world_camera,
                )
            )

    def stop(self):
        for cam in self._cameras:
            try:
                cam.pipeline.stop()
            except Exception as exc:
                print(f"[warn] Failed to stop RealSense pipeline for {cam.serial}: {exc}")

    @property
    def segmenter(self):
        return self._segmenter

    def capture_rgb(self, camera_index: int, warmup: int = 5) -> np.ndarray:
        if camera_index < 0 or camera_index >= len(self._cameras):
            raise IndexError("camera_index out of range")
        cam = self._cameras[camera_index]
        for _ in range(warmup):
            cam.pipeline.wait_for_frames()
        frames = cam.pipeline.wait_for_frames()
        if cam.align is not None:
            frames = cam.align.process(frames)
        color_frame = frames.get_color_frame()
        if not color_frame:
            raise RuntimeError(f"No color frame for camera index {camera_index}")
        return np.asanyarray(color_frame.get_data())

    def _segment(self, rgb: np.ndarray) -> Optional[np.ndarray]:
        if self._segmenter is None:
            return None
        return self._segmenter.segment(rgb)

    def capture_pcd_world(
        self,
        segmentation_masks: Optional[Iterable[np.ndarray]] = None,
        use_segmentation: bool = False,
        capture_debug_frames: bool = False,
    ) -> np.ndarray:
        all_points = []
        self._last_debug_frames = []
        segmenter_masks = None
        if use_segmentation and segmentation_masks is None and self._segmenter is None:
            raise RuntimeError(
                "Segmentation is enabled, but no segmentation source is available "
                "(no masks provided and no segmenter configured)."
            )
        if use_segmentation and segmentation_masks is None and self._segmenter is not None:
            if hasattr(self._segmenter, "get_masks"):
                segmenter_masks = self._segmenter.get_masks()
            else:
                segmenter_masks = None

        masks_iter = iter(segmentation_masks) if segmentation_masks is not None else None

        for idx, cam in enumerate(self._cameras):
            frames = cam.pipeline.wait_for_frames()
            if cam.align is not None:
                frames = cam.align.process(frames)
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                raise RuntimeError(
                    f"Missing RealSense frames for camera {cam.serial}: "
                    f"depth_ok={bool(depth_frame)} color_ok={bool(color_frame)}"
                )

            depth = np.asanyarray(depth_frame.get_data()).astype(np.float32) * cam.depth_scale
            color = np.asanyarray(color_frame.get_data())

            mask = next(masks_iter, None) if masks_iter is not None else None
            if mask is None and segmenter_masks is not None:
                if idx < len(segmenter_masks):
                    mask = segmenter_masks[idx]
            if mask is None and use_segmentation and self._segmenter is not None:
                if hasattr(self._segmenter, "segment_camera"):
                    mask = self._segmenter.segment_camera(color, idx)
                else:
                    mask = self._segment(color)
            if use_segmentation and mask is None:
                raise RuntimeError(
                    f"Segmentation is enabled but no mask was produced for camera {cam.serial} (index {idx})."
                )
            if mask is not None and mask.shape != depth.shape:
                raise ValueError(
                    f"Segmentation mask shape mismatch for camera {cam.serial}: "
                    f"mask={mask.shape}, depth={depth.shape}"
                )
            if mask is not None:
                depth = depth * mask.astype(np.float32)

            if capture_debug_frames:
                self._last_debug_frames.append(
                    {
                        "camera_index": idx,
                        "serial": cam.serial,
                        "rgb": color,
                        "mask": mask,
                    }
                )

            xyz_cam = _get_xyz(depth, cam.K).T
            valid = np.isfinite(xyz_cam).all(axis=1) & (xyz_cam[:, 2] > 0)
            xyz_cam = xyz_cam[valid]
            xyz_world = (cam.T_world_camera[:3, :3] @ xyz_cam.T).T + cam.T_world_camera[:3, 3]
            all_points.append(xyz_world)

        if not all_points:
            raise RuntimeError("No valid point clouds captured from any configured camera.")

        pcd = np.concatenate(all_points, axis=0)
        if self._voxel_size and o3d is not None:
            pcd = self._voxel_downsample(pcd, self._voxel_size)
        return pcd.astype(np.float32)

    def get_last_debug_frames(self):
        return list(self._last_debug_frames)

    @staticmethod
    def _voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points)
        down = cloud.voxel_down_sample(voxel_size)
        return np.asarray(down.points)
