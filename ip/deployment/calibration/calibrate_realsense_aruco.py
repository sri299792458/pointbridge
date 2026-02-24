#!/usr/bin/env python3
"""
Calibrate RealSense RGB camera(s) to a known ArUco marker pose in world frame.

Outputs T_world_camera for each serial using multi-sample averaging.
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pyrealsense2 as rs


def _aruco_dict(name: str):
    mapping = {
        "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
        "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
        "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
        "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
        "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
        "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
        "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
        "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
        "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
        "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
        "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
        "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
        "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
        "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
        "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
        "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
        "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
    }
    if name not in mapping:
        raise ValueError(f"Unknown ArUco dictionary: {name}")
    return cv2.aruco.getPredefinedDictionary(mapping[name])


def _marker_object_points(tag_size: float) -> np.ndarray:
    s = tag_size
    # Order must match ArUco corner order: tl, tr, br, bl
    return np.array(
        [
            [-s / 2, s / 2, 0.0],
            [s / 2, s / 2, 0.0],
            [s / 2, -s / 2, 0.0],
            [-s / 2, -s / 2, 0.0],
        ],
        dtype=np.float32,
    )


def _detect_marker_pose(
    bgr: np.ndarray,
    dictionary,
    parameters,
    K: np.ndarray,
    dist: np.ndarray,
    tag_id: int,
    tag_size: float,
) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(dictionary, parameters)
        corners, ids, _ = detector.detectMarkers(gray)
    else:
        corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary, parameters=parameters)

    if ids is None:
        return None
    ids = ids.flatten()
    if tag_id not in ids:
        return None
    idx = int(np.where(ids == tag_id)[0][0])
    marker_corners = corners[idx]

    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
        [marker_corners], tag_size, K, dist
    )
    rvec = rvecs[0, 0]
    tvec = tvecs[0, 0]

    obj_pts = _marker_object_points(tag_size)
    proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist)
    proj = proj.reshape(-1, 2)
    obs = marker_corners.reshape(-1, 2)
    reproj_err = float(np.linalg.norm(proj - obs, axis=1).mean())
    return rvec, tvec, reproj_err


def _rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    # Extrinsic XYZ (roll, pitch, yaw about world axes)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _quat_from_matrix(R: np.ndarray) -> np.ndarray:
    m00, m01, m02 = R[0]
    m10, m11, m12 = R[1]
    m20, m21, m22 = R[2]
    trace = m00 + m11 + m22
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m21 - m12) * s
        y = (m02 - m20) * s
        z = (m10 - m01) * s
    elif m00 > m11 and m00 > m22:
        s = 2.0 * np.sqrt(1.0 + m00 - m11 - m22)
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = 2.0 * np.sqrt(1.0 + m11 - m00 - m22)
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m22 - m00 - m11)
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    if q[0] < 0:
        q = -q
    return q


def _matrix_from_quat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    ww = w * w
    xx = x * x
    yy = y * y
    zz = z * z
    wx = w * x
    wy = w * y
    wz = w * z
    xy = x * y
    xz = x * z
    yz = y * z
    return np.array(
        [
            [ww + xx - yy - zz, 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), ww - xx + yy - zz, 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), ww - xx - yy + zz],
        ],
        dtype=np.float64,
    )


def _average_rotations(rotations: List[np.ndarray]) -> np.ndarray:
    A = np.zeros((4, 4), dtype=np.float64)
    for R in rotations:
        q = _quat_from_matrix(R)
        A += np.outer(q, q)
    eigvals, eigvecs = np.linalg.eigh(A)
    q_mean = eigvecs[:, np.argmax(eigvals)]
    if q_mean[0] < 0:
        q_mean = -q_mean
    return _matrix_from_quat(q_mean)


def _transform_from_rvec_tvec(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = tvec
    return T


def _rotation_angle_deg(R: np.ndarray) -> float:
    trace = np.clip((np.trace(R) - 1) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(trace)))


def _start_pipeline(serial: str, width: int, height: int, fps: int):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    profile = pipeline.start(config)
    stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = stream.get_intrinsics()
    K = np.array(
        [[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]],
        dtype=np.float64,
    )
    dist = np.array(intr.coeffs, dtype=np.float64)
    return pipeline, K, dist


def _calibrate_camera(
    serial: str,
    tag_id: int,
    tag_size: float,
    dictionary,
    parameters,
    T_world_tag: np.ndarray,
    width: int,
    height: int,
    fps: int,
    warmup_frames: int,
    num_samples: int,
    max_frames: int,
    max_reproj_error: float,
) -> Dict:
    pipeline, K, dist = _start_pipeline(serial, width, height, fps)
    try:
        for _ in range(warmup_frames):
            pipeline.wait_for_frames()

        rotations = []
        translations = []
        reproj_errors = []
        used_frames = 0

        while len(rotations) < num_samples and used_frames < max_frames:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                used_frames += 1
                continue
            bgr = np.asanyarray(color_frame.get_data())

            pose = _detect_marker_pose(bgr, dictionary, parameters, K, dist, tag_id, tag_size)
            used_frames += 1
            if pose is None:
                continue
            rvec, tvec, reproj_err = pose
            if reproj_err > max_reproj_error:
                continue

            T_cam_tag = _transform_from_rvec_tvec(rvec, tvec)
            T_tag_cam = np.linalg.inv(T_cam_tag)
            T_world_cam = T_world_tag @ T_tag_cam

            rotations.append(T_world_cam[:3, :3])
            translations.append(T_world_cam[:3, 3])
            reproj_errors.append(reproj_err)

        if not rotations:
            raise RuntimeError(
                f"No valid detections for serial {serial}. "
                "Check tag visibility, size, and dictionary."
            )

        R_mean = _average_rotations(rotations)
        t_mean = np.mean(np.stack(translations, axis=0), axis=0)
        T_mean = np.eye(4, dtype=np.float64)
        T_mean[:3, :3] = R_mean
        T_mean[:3, 3] = t_mean

        rot_errors = [
            _rotation_angle_deg(R_mean.T @ R_i) for R_i in rotations
        ]
        return {
            "serial": serial,
            "T_world_camera": T_mean,
            "num_samples": len(rotations),
            "used_frames": used_frames,
            "reproj_error_mean_px": float(np.mean(reproj_errors)),
            "reproj_error_std_px": float(np.std(reproj_errors)),
            "rot_error_mean_deg": float(np.mean(rot_errors)),
            "rot_error_std_deg": float(np.std(rot_errors)),
            "translation_std_m": np.std(np.stack(translations, axis=0), axis=0).tolist(),
        }
    finally:
        try:
            pipeline.stop()
        except Exception as exc:
            print(f"[warn] Failed to stop pipeline for serial {serial}: {exc}")


def _default_world_tag_path(arm: str) -> Path:
    filename = "world_tag_right.json" if arm == "right" else "world_tag.json"
    return Path(__file__).resolve().parent / "outputs" / filename


def _default_out_path(arm: str) -> Path:
    filename = "realsense_T_world_camera_right.json" if arm == "right" else "realsense_T_world_camera.json"
    return Path(__file__).resolve().parent / "outputs" / filename


def _parse_world_tag(args) -> np.ndarray:
    if args.world_tag_matrix:
        path = Path(args.world_tag_matrix)
    elif args.world_tag is None:
        path = _default_world_tag_path(args.arm)
        if not path.exists():
            raise ValueError(
                f"Provide --world-tag or --world-tag-matrix. Default for --arm {args.arm!r} "
                f"not found: {path}"
            )
        print(f"Using default world-tag matrix for --arm {args.arm}: {path}")
    else:
        path = None

    if path is not None:
        if path.suffix.lower() in {".npy", ".npz"}:
            T = np.load(path)
        else:
            with path.open("r", encoding="utf-8") as f:
                T = np.array(json.load(f), dtype=np.float64)
        T = np.array(T, dtype=np.float64)
        if T.shape != (4, 4):
            raise ValueError("world_tag_matrix must be 4x4")
        return T

    x, y, z, roll_deg, pitch_deg, yaw_deg = args.world_tag
    R = _rpy_to_matrix(
        np.deg2rad(roll_deg), np.deg2rad(pitch_deg), np.deg2rad(yaw_deg)
    )
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = np.array([x, y, z], dtype=np.float64)
    return T


def main():
    if not hasattr(cv2, "aruco"):
        raise ImportError("cv2.aruco is missing. Install opencv-contrib-python.")
    parser = argparse.ArgumentParser(
        description="Calibrate RealSense camera(s) to ArUco marker pose in world frame."
    )
    parser.add_argument(
        "--arm",
        choices=["left", "right"],
        default="left",
        help="Arm side used for default world-tag and output file naming.",
    )
    parser.add_argument("--serial", action="append", required=True, help="RealSense serial")
    parser.add_argument("--tag-dict", default="DICT_4X4_50", help="ArUco dictionary name")
    parser.add_argument("--tag-id", type=int, default=1, help="ArUco marker ID")
    parser.add_argument("--tag-size", type=float, required=True, help="Marker edge length (meters)")
    parser.add_argument(
        "--world-tag",
        type=float,
        nargs=6,
        metavar=("X", "Y", "Z", "ROLL", "PITCH", "YAW"),
        help="Tag pose in world: meters + degrees (roll/pitch/yaw about world axes)",
    )
    parser.add_argument(
        "--world-tag-matrix",
        type=str,
        default=None,
        help="Path to 4x4 world->tag matrix (.json or .npy)",
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--warmup-frames", type=int, default=30)
    parser.add_argument("--num-samples", type=int, default=30)
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--max-reproj-error", type=float, default=2.0)
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output JSON path (default depends on --arm).",
    )
    args = parser.parse_args()

    T_world_tag = _parse_world_tag(args)
    dictionary = _aruco_dict(args.tag_dict)
    if hasattr(cv2.aruco, "DetectorParameters"):
        parameters = cv2.aruco.DetectorParameters()
    else:
        parameters = cv2.aruco.DetectorParameters_create()

    results = {
        "tag": {
            "dict": args.tag_dict,
            "id": args.tag_id,
            "size_m": args.tag_size,
            "T_world_tag": T_world_tag.tolist(),
        },
        "cameras": {},
    }

    for serial in args.serial:
        stats = _calibrate_camera(
            serial=serial,
            tag_id=args.tag_id,
            tag_size=args.tag_size,
            dictionary=dictionary,
            parameters=parameters,
            T_world_tag=T_world_tag,
            width=args.width,
            height=args.height,
            fps=args.fps,
            warmup_frames=args.warmup_frames,
            num_samples=args.num_samples,
            max_frames=args.max_frames,
            max_reproj_error=args.max_reproj_error,
        )
        results["cameras"][serial] = {
            "T_world_camera": stats["T_world_camera"].tolist(),
            "num_samples": stats["num_samples"],
            "used_frames": stats["used_frames"],
            "reproj_error_mean_px": stats["reproj_error_mean_px"],
            "reproj_error_std_px": stats["reproj_error_std_px"],
            "rot_error_mean_deg": stats["rot_error_mean_deg"],
            "rot_error_std_deg": stats["rot_error_std_deg"],
            "translation_std_m": stats["translation_std_m"],
        }

        print(f"\nSerial {serial} T_world_camera:")
        print(np.array(results["cameras"][serial]["T_world_camera"]))
        print(
            "  samples:",
            stats["num_samples"],
            "reproj_err(px):",
            f"{stats['reproj_error_mean_px']:.2f}±{stats['reproj_error_std_px']:.2f}",
            "rot_err(deg):",
            f"{stats['rot_error_mean_deg']:.2f}±{stats['rot_error_std_deg']:.2f}",
        )

    if args.out:
        out_path = Path(args.out)
    else:
        out_path = _default_out_path(args.arm)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved calibration to {out_path}")


if __name__ == "__main__":
    main()
