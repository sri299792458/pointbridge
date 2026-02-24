from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from ip.deployment.control.action_executor import SafetyLimits


@dataclass
class CameraConfig:
    serial: str
    T_world_camera: np.ndarray
    width: int = 640
    height: int = 480
    fps: int = 30
    align_to_color: bool = True


@dataclass
class SegmentationConfig:
    enable: bool = True
    backend: str = "xmem"
    model_type: str = "vit_b"
    checkpoint_path: Optional[str] = None
    sam_checkpoint_path: Optional[str] = None
    xmem_checkpoint_path: Optional[str] = None
    xmem_init_with_sam: bool = True
    xmem_config_overrides: Optional[dict] = None
    points_per_side: int = 32
    pred_iou_thresh: float = 0.88
    stability_score_thresh: float = 0.95
    min_mask_region_area: int = 256
    select_largest: bool = True


@dataclass
class GripperConfig:
    enable: bool = True
    host: Optional[str] = None
    port: int = 63352
    open_position: int = 0
    closed_position: int = 255
    speed: int = 255
    force: int = 100


@dataclass
class RTDEControlConfig:
    frequency_hz: int = 500
    control_mode: str = "servoL"  # moveL or servoL
    move_speed: float = 0.25
    move_acceleration: float = 1.2
    # NOTE: For UR servoJ/servoL in RTDE, speed/acceleration are currently not used.
    servo_speed: float = 0.25
    servo_acceleration: float = 1.2
    # e-Series default servo period is 2ms.
    servo_time: float = 0.002
    servo_lookahead: float = 0.1
    servo_gain: int = 300


@dataclass
class DeploymentConfig:
    camera_configs: List[CameraConfig] = field(default_factory=list)
    robot_ip: str = "192.168.1.102"
    model_path: str = "./checkpoints/ip"
    num_demos: int = 2
    num_traj_wp: int = 10
    max_execution_steps: int = 100
    num_diffusion_iters: int = 8
    pcd_num_points: int = 2048
    pcd_voxel_size: Optional[float] = None
    safety: Optional[SafetyLimits] = None
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    rtde: RTDEControlConfig = field(default_factory=RTDEControlConfig)
    gripper: GripperConfig = field(default_factory=GripperConfig)
    device: Optional[str] = None
    execute_until_grip_change: bool = True
    # Runtime convention:
    # - RTDE TCP is flange.
    # - Policy pose frame is flange translated by this fixed offset.
    tcp_offset_in_code: bool = True
    tcp_offset_m: np.ndarray = field(default_factory=lambda: np.array([0.000, 0.000, 0.088], dtype=np.float64))
    debug_frame_sanity: bool = False
    debug_frame_every: int = 1
