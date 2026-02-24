from typing import Optional

import numpy as np
import rtde_receive
from scipy.spatial.transform import Rotation

from ip.deployment.control.robotiq_gripper import RobotiqGripper


class URRTDEState:
    def __init__(
        self,
        rtde: "rtde_receive.RTDEReceiveInterface",
        gripper: Optional[RobotiqGripper] = None,
        tcp_offset_in_code: bool = False,
        tcp_offset_m: Optional[np.ndarray] = None,
    ):
        self._rtde = rtde
        self._gripper = gripper
        self._tcp_offset_in_code = tcp_offset_in_code
        self._tcp_offset_m = tcp_offset_m

    @staticmethod
    def connect(robot_ip: str) -> "rtde_receive.RTDEReceiveInterface":
        return rtde_receive.RTDEReceiveInterface(robot_ip)

    def get_T_w_e(self) -> np.ndarray:
        pose = self._rtde.getActualTCPPose()
        T = np.eye(4)
        T[:3, 3] = pose[:3]
        T[:3, :3] = Rotation.from_rotvec(pose[3:]).as_matrix()
        if self._tcp_offset_in_code and self._tcp_offset_m is not None:
            T_offset = np.eye(4)
            T_offset[:3, 3] = self._tcp_offset_m
            T = T @ T_offset
        return T

    def get_actual_q(self) -> np.ndarray:
        q = np.asarray(self._rtde.getActualQ(), dtype=np.float64)
        if q.shape != (6,):
            raise RuntimeError(f"Unexpected joint vector shape from RTDE: {q.shape}")
        if not np.isfinite(q).all():
            raise RuntimeError(f"Non-finite joint values from RTDE: {q.tolist()}")
        return q

    def get_gripper_state(self) -> float:
        if self._gripper is None:
            raise RuntimeError("Robotiq gripper is not available for gripper state feedback.")
        pos = float(self._gripper.get_position_normalized())
        if not np.isfinite(pos):
            raise RuntimeError(f"Non-finite Robotiq normalized position: {pos}")
        # Map Robotiq normalized position (0=open, 1=closed) to model convention (1=open, 0=closed).
        return float(1.0 - pos)

    def get_gripper_obj_state(self, require: bool = False) -> Optional[int]:
        if self._gripper is None:
            if require:
                raise RuntimeError("Robotiq gripper is not available for OBJ feedback.")
            return None
        return int(self._gripper.get_object_status())
