import logging
from typing import Optional

import numpy as np
import torch
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry


class SAMSegmenter:
    def __init__(
        self,
        model_type: str,
        checkpoint_path: str,
        device: str,
        points_per_side: int = 32,
        pred_iou_thresh: float = 0.88,
        stability_score_thresh: float = 0.95,
        min_mask_region_area: int = 256,
        select_largest: bool = True,
    ):
        if not checkpoint_path:
            raise ValueError("SAM checkpoint_path is required")
        if model_type not in sam_model_registry:
            raise ValueError(f"Unknown SAM model type: {model_type}")

        sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        sam.to(device=device)
        self._generator = SamAutomaticMaskGenerator(
            sam,
            points_per_side=points_per_side,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
            min_mask_region_area=min_mask_region_area,
        )
        self._select_largest = select_largest

    def segment(self, rgb: np.ndarray) -> Optional[np.ndarray]:
        masks = self._generator.generate(rgb)
        if not masks:
            return None
        if self._select_largest:
            best = max(masks, key=lambda m: m.get("area", m["segmentation"].sum()))
            return best["segmentation"].astype(np.uint8)
        combined = np.zeros_like(masks[0]["segmentation"], dtype=bool)
        for m in masks:
            combined |= m["segmentation"]
        return combined.astype(np.uint8)

    def segment_camera(self, rgb: np.ndarray, camera_index: int) -> Optional[np.ndarray]:
        return self.segment(rgb)


def build_segmenter(config, device: Optional[str] = None, num_cameras: Optional[int] = None):
    if config is None or not getattr(config, "enable", False):
        return None
    backend = getattr(config, "backend", "sam").lower()
    if backend == "xmem":
        from ip.deployment.perception.xmem_segmentation import XMemOnlineSegmenter
        if getattr(config, "xmem_init_with_sam", True):
            checkpoint = config.sam_checkpoint_path or config.checkpoint_path
            if not checkpoint:
                raise ValueError("SAM checkpoint is required to seed XMem++")
        if num_cameras is None:
            raise ValueError("num_cameras is required for XMem online segmentation")
        return XMemOnlineSegmenter(
            num_cameras=num_cameras,
            checkpoint_path=config.xmem_checkpoint_path or config.checkpoint_path,
            device=device,
            init_with_sam=config.xmem_init_with_sam,
            sam_config=config,
            config_overrides=config.xmem_config_overrides,
        )
    if backend != "sam":
        raise ValueError(f"Unknown segmentation backend: {backend}")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return SAMSegmenter(
        model_type=config.model_type,
        checkpoint_path=config.sam_checkpoint_path or config.checkpoint_path,
        device=device,
        points_per_side=config.points_per_side,
        pred_iou_thresh=config.pred_iou_thresh,
        stability_score_thresh=config.stability_score_thresh,
        min_mask_region_area=config.min_mask_region_area,
        select_largest=config.select_largest,
    )
