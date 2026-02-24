from typing import Optional

import numpy as np
import torch


def _import_xmem_modules():
    try:
        from model.network import XMem
        from util.configuration import VIDEO_INFERENCE_CONFIG
        from dataset.range_transform import im_normalization
        from inference.inference_core import InferenceCore
        return XMem, VIDEO_INFERENCE_CONFIG, im_normalization, InferenceCore
    except Exception as exc:
        raise ImportError(
            "XMem2 modules are not importable. Add XMem2 root to PYTHONPATH "
            "(for example with a .pth file in your active environment)."
        ) from exc


class XMemOnlineSegmenter:
    def __init__(
        self,
        num_cameras: int,
        checkpoint_path: Optional[str],
        device: Optional[str] = None,
        init_with_sam: bool = True,
        sam_config=None,
        config_overrides: Optional[dict] = None,
    ):
        XMem, VIDEO_INFERENCE_CONFIG, im_normalization, InferenceCore = _import_xmem_modules()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        device_str = str(device)
        if not device_str.startswith("cuda"):
            raise RuntimeError("XMem++ requires CUDA for real-time use")
        if device_str not in ("cuda", "cuda:0"):
            raise RuntimeError("XMem++ inference expects device cuda:0")
        if device_str == "cuda":
            device = "cuda:0"

        if not checkpoint_path:
            raise ValueError("XMem++ checkpoint_path is required")

        self._device = torch.device(device)
        self._im_normalization = im_normalization
        self._num_cameras = num_cameras
        self._initialized = [False] * num_cameras

        config = VIDEO_INFERENCE_CONFIG.copy()
        config["model"] = checkpoint_path
        config["size"] = -1
        config["save_masks"] = False
        if config_overrides:
            config.update(config_overrides)
        self._config = config

        self._network = XMem(
            config,
            checkpoint_path,
            pretrained_key_encoder=False,
            pretrained_value_encoder=False,
        ).to(self._device).eval()

        self._processors = [InferenceCore(self._network, config) for _ in range(num_cameras)]
        self._labels = [1]

        self._sam = None
        if init_with_sam:
            if sam_config is None:
                raise ValueError("SAM config is required to seed XMem++")
            from ip.deployment.perception.sam_segmentation import SAMSegmenter
            checkpoint = sam_config.sam_checkpoint_path or sam_config.checkpoint_path
            if not checkpoint:
                raise ValueError("SAM checkpoint is required to seed XMem++")
            self._sam = SAMSegmenter(
                model_type=sam_config.model_type,
                checkpoint_path=checkpoint,
                device=device,
                points_per_side=sam_config.points_per_side,
                pred_iou_thresh=sam_config.pred_iou_thresh,
                stability_score_thresh=sam_config.stability_score_thresh,
                min_mask_region_area=sam_config.min_mask_region_area,
                select_largest=sam_config.select_largest,
            )

        self._torch = torch

    def segment_camera(self, rgb: np.ndarray, camera_index: int) -> np.ndarray:
        if camera_index >= self._num_cameras:
            raise IndexError(
                f"camera_index {camera_index} out of range for XMem segmenter with {self._num_cameras} cameras."
            )

        if not self._initialized[camera_index]:
            if self._sam is None:
                raise RuntimeError(
                    f"XMem camera {camera_index} is not initialized and SAM seeding is disabled. "
                    "Run manual seeding first."
                )
            mask = self._sam.segment(rgb)
            if mask is None or mask.sum() == 0:
                raise RuntimeError(
                    f"SAM failed to produce a non-empty seed mask for XMem camera {camera_index}."
                )
            self._initialize(camera_index, rgb, mask)
            return mask.astype(np.uint8)

        return self._track(camera_index, rgb)

    def initialize_camera(self, camera_index: int, rgb: np.ndarray, mask: np.ndarray) -> None:
        if camera_index >= self._num_cameras:
            raise IndexError(
                f"camera_index {camera_index} out of range for XMem segmenter with {self._num_cameras} cameras."
            )
        self._initialize(camera_index, rgb, mask)

    def _initialize(self, camera_index: int, rgb: np.ndarray, mask: np.ndarray) -> None:
        processor = self._processors[camera_index]
        processor.clear_memory()
        processor.set_all_labels(self._labels)

        image_t = self._prepare_image(rgb)
        mask_t = self._prepare_mask(mask)
        with self._torch.no_grad():
            processor.put_to_permanent_memory(image_t, mask_t, ti=0)
        self._initialized[camera_index] = True

    def _track(self, camera_index: int, rgb: np.ndarray) -> np.ndarray:
        processor = self._processors[camera_index]
        image_t = self._prepare_image(rgb)
        with self._torch.no_grad():
            prob = processor.step(image_t, mask=None, valid_labels=None)
        if prob is None:
            raise RuntimeError(f"XMem tracking returned no probabilities for camera {camera_index}.")
        pred = self._torch.argmax(prob, dim=0).detach().cpu().numpy().astype(np.uint8)
        return (pred > 0).astype(np.uint8)

    def _prepare_image(self, rgb: np.ndarray):
        image = self._torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        image = self._im_normalization(image)
        return image.to(self._device)

    def _prepare_mask(self, mask: np.ndarray):
        if mask.dtype != np.float32:
            mask = mask.astype(np.float32)
        if mask.max() > 1.0:
            mask = (mask > 0).astype(np.float32)
        mask_t = self._torch.from_numpy(mask)
        if mask_t.dim() == 2:
            mask_t = mask_t.unsqueeze(0)
        return mask_t.to(self._device)
