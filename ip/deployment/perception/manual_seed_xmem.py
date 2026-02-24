#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np

from ip.deployment.perception.realsense_perception import RealSensePerception


def _interactive_grabcut(rgb: np.ndarray, window_name: str) -> np.ndarray:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    combined_mask = np.zeros(bgr.shape[:2], np.uint8)
    while True:
        print(f"[{window_name}] Draw a box around an object and press Enter/Space (Esc to finish).")
        rect = cv2.selectROI(window_name, bgr, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow(window_name)
        x, y, w, h = [int(v) for v in rect]
        if w <= 0 or h <= 0:
            if combined_mask.sum() > 0:
                return combined_mask
            print("Empty selection. Try again.")
            continue

        mask = np.zeros(bgr.shape[:2], np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        cv2.grabCut(bgr, mask, (x, y, w, h), bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
        mask_bin = np.where(
            (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
            1,
            0,
        ).astype(np.uint8)

        candidate = np.clip(combined_mask + mask_bin, 0, 1).astype(np.uint8)
        overlay = bgr.copy()
        overlay[candidate == 0] = (overlay[candidate == 0] * 0.2).astype(np.uint8)
        cv2.imshow("Mask preview (y=add, n=finish, r=redo, c=clear)", overlay)
        key = cv2.waitKey(0) & 0xFF
        cv2.destroyWindow("Mask preview (y=add, n=finish, r=redo, c=clear)")

        if key in (ord("y"), ord("Y")):
            combined_mask = candidate
            print("Added. Draw another object.")
            continue
        if key in (ord("n"), ord("N")):
            combined_mask = candidate
            return combined_mask
        if key in (ord("c"), ord("C")):
            combined_mask = np.zeros(bgr.shape[:2], np.uint8)
            print("Cleared all masks. Start again.")
            continue
        print("Redo selection...")


def manual_seed_xmem(
    perception: RealSensePerception,
    camera_serials: Iterable[str],
    out_dir: Optional[str] = None,
    warmup: int = 5,
) -> None:
    segmenter = getattr(perception, "segmenter", None)
    if segmenter is None or not hasattr(segmenter, "initialize_camera"):
        raise RuntimeError("Segmenter does not support manual initialization")

    out_path = Path(out_dir) if out_dir else None
    if out_path:
        out_path.mkdir(parents=True, exist_ok=True)

    serials = list(camera_serials)
    for idx, serial in enumerate(serials):
        rgb = perception.capture_rgb(idx, warmup=warmup)
        mask = _interactive_grabcut(rgb, f"Camera {serial}")
        segmenter.initialize_camera(idx, rgb, mask)

        if out_path:
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            overlay = bgr.copy()
            overlay[mask == 0] = (overlay[mask == 0] * 0.2).astype(np.uint8)
            cv2.imwrite(str(out_path / f"{serial}_color.png"), bgr)
            cv2.imwrite(str(out_path / f"{serial}_mask.png"), (mask * 255).astype(np.uint8))
            cv2.imwrite(str(out_path / f"{serial}_overlay.png"), overlay)


def main():
    parser = argparse.ArgumentParser(description="Manually seed XMem++ masks using GrabCut.")
    parser.add_argument("--out-dir", default=None, help="Optional output directory for saved masks")
    args = parser.parse_args()
    raise RuntimeError(
        "This module is intended to be called from ip.deployment with --manual-seed. "
        "Run: python -m ip.deployment --manual-seed --demo <demo.pkl>"
    )


if __name__ == "__main__":
    main()
