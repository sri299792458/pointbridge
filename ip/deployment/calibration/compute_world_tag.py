#!/usr/bin/env python3
"""
Compute T_world_tag from four measured ArUco corner points (TL, TR, BR, BL).
Inputs are four flange poses (RTDE TCP = flange), each as:
  - x y z rx ry rz

Each flange pose is converted to a contact point using flange->contact offset.

Tag frame convention matches OpenCV ArUco:
  - Origin at tag center
  - X axis: left -> right (TL -> TR)
  - Y axis: bottom -> top (BL -> TL)
  - Z axis: right-handed (X x Y), out of tag plane
"""
import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


def _parse_vec(values: Tuple[float, ...], name: str) -> np.ndarray:
    if len(values) != 6:
        raise ValueError(f"{name} must have 6 values: x y z rx ry rz")
    return np.array(values, dtype=np.float64)


def _normalize(v: np.ndarray, name: str) -> np.ndarray:
    n = np.linalg.norm(v)
    if n <= 0:
        raise ValueError(f"{name} has zero length")
    return v / n


def _rotvec_to_matrix(rotvec: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(rotvec))
    if theta <= 0:
        return np.eye(3, dtype=np.float64)
    k = rotvec / theta
    kx, ky, kz = k
    K = np.array(
        [[0.0, -kz, ky], [kz, 0.0, -kx], [-ky, kx, 0.0]],
        dtype=np.float64,
    )
    return np.eye(3, dtype=np.float64) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def _to_contact_position(values: np.ndarray, flange_to_contact_m: np.ndarray) -> np.ndarray:
    pos = values[:3]
    rotvec = values[3:]
    R = _rotvec_to_matrix(rotvec)
    return pos + (R @ flange_to_contact_m)


def compute_T_world_tag_best_fit(
    tl: np.ndarray,
    tr: np.ndarray,
    br: np.ndarray,
    bl: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, float]]:
    points = np.stack([tl, tr, br, bl], axis=0)
    center = points.mean(axis=0)
    centered = points - center

    # Best-fit tag plane from all four points.
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    plane_normal = _normalize(vh[-1], "plane_normal")

    # Average opposite edges to reduce measurement noise.
    x_raw = 0.5 * ((tr - tl) + (br - bl))
    y_raw = 0.5 * ((tl - bl) + (tr - br))

    # Force in-plane axes.
    x_in_plane = x_raw - np.dot(x_raw, plane_normal) * plane_normal
    y_in_plane = y_raw - np.dot(y_raw, plane_normal) * plane_normal

    x_axis = _normalize(x_in_plane, "x_axis")
    y_in_plane = y_in_plane - np.dot(y_in_plane, x_axis) * x_axis
    y_axis = _normalize(y_in_plane, "y_axis")
    z_axis = _normalize(np.cross(x_axis, y_axis), "z_axis")

    # Keep right-handed frame while aligning z with plane normal direction.
    if float(np.dot(z_axis, plane_normal)) < 0.0:
        y_axis = -y_axis
        z_axis = -z_axis

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.stack([x_axis, y_axis, z_axis], axis=1)
    T[:3, 3] = center

    plane_dist = np.abs(centered @ z_axis)
    stats = {
        "plane_fit_rms_m": float(np.sqrt(np.mean(plane_dist ** 2))),
        "plane_fit_max_m": float(np.max(plane_dist)),
    }
    return T, stats


def _default_out_path(arm: str) -> Path:
    filename = "world_tag_right.json" if arm == "right" else "world_tag.json"
    return Path(__file__).resolve().parent / "outputs" / filename


def main():
    parser = argparse.ArgumentParser(description="Compute T_world_tag from four ArUco corners (best-fit).")
    parser.add_argument(
        "--arm",
        choices=["left", "right"],
        default="left",
        help="Arm side used for default output naming.",
    )
    parser.add_argument(
        "--tl",
        type=float,
        nargs=6,
        required=True,
        metavar=("X", "Y", "Z", "RX", "RY", "RZ"),
        help="Top-left corner flange pose (x y z rx ry rz).",
    )
    parser.add_argument(
        "--tr",
        type=float,
        nargs=6,
        required=True,
        metavar=("X", "Y", "Z", "RX", "RY", "RZ"),
        help="Top-right corner flange pose (x y z rx ry rz).",
    )
    parser.add_argument(
        "--bl",
        type=float,
        nargs=6,
        required=True,
        metavar=("X", "Y", "Z", "RX", "RY", "RZ"),
        help="Bottom-left corner flange pose (x y z rx ry rz).",
    )
    parser.add_argument(
        "--br",
        type=float,
        nargs=6,
        required=True,
        metavar=("X", "Y", "Z", "RX", "RY", "RZ"),
        help="Bottom-right corner flange pose (x y z rx ry rz).",
    )
    parser.add_argument(
        "--flange-to-contact-m",
        "--tcp-offset-m",
        dest="flange_to_contact_m",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.162),
        metavar=("X", "Y", "Z"),
        help="Offset from flange to contact point in meters.",
    )
    parser.add_argument(
        "--tag-size",
        type=float,
        default=None,
        help="Optional: tag edge length (meters) for sanity check.",
    )
    parser.add_argument(
        "--warn-threshold",
        type=float,
        default=0.005,
        help="Warn if measured edge differs from tag size by more than this (meters).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output JSON path for T_world_tag (default depends on --arm).",
    )
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="Only print the matrix; do not write a file.",
    )
    args = parser.parse_args()

    tl_raw = _parse_vec(tuple(args.tl), "tl")
    tr_raw = _parse_vec(tuple(args.tr), "tr")
    br_raw = _parse_vec(tuple(args.br), "br")
    bl_raw = _parse_vec(tuple(args.bl), "bl")
    flange_to_contact = np.array(args.flange_to_contact_m, dtype=np.float64)
    print("Interpreting inputs as flange poses (x y z rx ry rz).")
    print(f"Applying flange->contact offset (m): {flange_to_contact.tolist()}")

    tl = _to_contact_position(tl_raw, flange_to_contact)
    tr = _to_contact_position(tr_raw, flange_to_contact)
    br = _to_contact_position(br_raw, flange_to_contact)
    bl = _to_contact_position(bl_raw, flange_to_contact)

    T, fit_stats = compute_T_world_tag_best_fit(tl, tr, br, bl)

    edge_top = float(np.linalg.norm(tr - tl))
    edge_right = float(np.linalg.norm(tr - br))
    edge_bottom = float(np.linalg.norm(br - bl))
    edge_left = float(np.linalg.norm(tl - bl))
    print(
        "Measured edge lengths (m): "
        f"top={edge_top:.6f}, right={edge_right:.6f}, bottom={edge_bottom:.6f}, left={edge_left:.6f}"
    )
    print(
        "Plane fit residuals (m): "
        f"rms={fit_stats['plane_fit_rms_m']:.6f}, max={fit_stats['plane_fit_max_m']:.6f}"
    )
    if args.tag_size is not None:
        edge_errors = [
            abs(edge_top - args.tag_size),
            abs(edge_right - args.tag_size),
            abs(edge_bottom - args.tag_size),
            abs(edge_left - args.tag_size),
        ]
        if any(err > args.warn_threshold for err in edge_errors):
            print(
                f"WARNING: Edge mismatch vs tag_size={args.tag_size:.6f} m "
                f"(dt={edge_errors[0]:.6f}, dr={edge_errors[1]:.6f}, "
                f"db={edge_errors[2]:.6f}, dl={edge_errors[3]:.6f})."
            )

    print("T_world_tag:")
    for row in T:
        print(row.tolist())

    if args.print_only:
        return

    out_path = Path(args.out) if args.out else _default_out_path(args.arm)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(T.tolist(), f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
