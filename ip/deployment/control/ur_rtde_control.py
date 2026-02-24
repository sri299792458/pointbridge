import time
from typing import Optional, Tuple

import numpy as np
import rtde_control
from scipy.spatial.transform import Rotation

from ip.deployment.config import GripperConfig, RTDEControlConfig
from ip.deployment.control.robotiq_gripper import RobotiqGripper


class URRTDEControl:
    def __init__(
        self,
        rtde: "rtde_control.RTDEControlInterface",
        control_config: RTDEControlConfig,
        gripper: Optional[RobotiqGripper] = None,
        gripper_config: Optional[GripperConfig] = None,
        tcp_offset_in_code: bool = False,
        tcp_offset_m: Optional[np.ndarray] = None,
    ):
        self._rtde = rtde
        self._cfg = control_config
        self._mode = self._cfg.control_mode.lower()
        if self._mode not in {"servol", "movel"}:
            raise ValueError(
                f"Unsupported control_mode={self._cfg.control_mode!r}. Expected 'servoL' or 'moveL'."
            )
        self._gripper = gripper
        self._gripper_cfg = gripper_config or GripperConfig(enable=gripper is not None)
        if self._gripper_cfg.enable and self._gripper is None:
            raise ValueError("Gripper is enabled but no RobotiqGripper instance was provided.")
        self._tcp_offset_in_code = tcp_offset_in_code
        self._tcp_offset_m = tcp_offset_m

    @staticmethod
    def connect(robot_ip: str, control_config: RTDEControlConfig) -> "rtde_control.RTDEControlInterface":
        return rtde_control.RTDEControlInterface(robot_ip, control_config.frequency_hz)

    def _to_rtde_pose(self, T_w_e: np.ndarray) -> list:
        if self._tcp_offset_in_code and self._tcp_offset_m is not None:
            T_offset = np.eye(4, dtype=np.float64)
            T_offset[:3, 3] = self._tcp_offset_m
            T_w_e = T_w_e @ np.linalg.inv(T_offset)
        position = T_w_e[:3, 3]
        rotvec = Rotation.from_matrix(T_w_e[:3, :3]).as_rotvec()
        return list(position) + list(rotvec)

    @staticmethod
    def _is_command_success(result) -> bool:
        # ur_rtde methods usually return bool; treat only explicit False as failure.
        return result is not False

    def validate_target_pose(self, T_w_e: np.ndarray, q_near: np.ndarray) -> Tuple[bool, str]:
        pose = self._to_rtde_pose(T_w_e)
        q_near_arr = np.asarray(q_near, dtype=np.float64)
        if q_near_arr.shape != (6,):
            raise RuntimeError(f"q_near must be shape (6,), got {q_near_arr.shape}")
        q_near_list = [float(x) for x in q_near_arr.tolist()]

        if not self._rtde.isPoseWithinSafetyLimits(pose):
            return False, "Target pose violates UR safety limits"
        if not self._rtde.getInverseKinematicsHasSolution(pose, q_near_list):
            return False, "No IK solution for target pose near current joints"

        q_sol = self._rtde.getInverseKinematics(pose, q_near_list)
        if len(q_sol) != 6:
            return False, "IK solver did not return a 6-DoF joint solution"
        if not self._rtde.isJointsWithinSafetyLimits(q_sol):
            return False, "IK solution violates UR joint safety limits"
        return True, ""

    def execute_pose(self, T_w_e: np.ndarray) -> bool:
        pose = self._to_rtde_pose(T_w_e)

        if self._mode == "servol":
            result = self._rtde.servoL(
                pose,
                self._cfg.servo_speed,
                self._cfg.servo_acceleration,
                self._cfg.servo_time,
                self._cfg.servo_lookahead,
                self._cfg.servo_gain,
            )
            if self._cfg.servo_time > 0:
                time.sleep(self._cfg.servo_time)
            return self._is_command_success(result)
        else:
            result = self._rtde.moveL(
                pose,
                self._cfg.move_speed,
                self._cfg.move_acceleration,
            )
            return self._is_command_success(result)

    def execute_joint_positions(self, joints_rad: list[float]) -> bool:
        if len(joints_rad) != 6:
            raise ValueError(f"Expected 6 joint values, got {len(joints_rad)}")
        result = self._rtde.servoJ(
            [float(x) for x in joints_rad],
            0.0,
            0.0,
            self._cfg.servo_time,
            self._cfg.servo_lookahead,
            self._cfg.servo_gain,
        )
        if self._cfg.servo_time > 0:
            time.sleep(self._cfg.servo_time)
        return self._is_command_success(result)

    def execute_gripper(self, command: float) -> None:
        if not self._gripper_cfg.enable:
            return
        if self._gripper is None:
            raise RuntimeError("Gripper command requested but RobotiqGripper is not available.")
        # Convention: command >= 0.5 means OPEN, < 0.5 means CLOSED.
        if command > 0.5:
            self._gripper.open(speed=self._gripper_cfg.speed, force=self._gripper_cfg.force)
        else:
            self._gripper.close(speed=self._gripper_cfg.speed, force=self._gripper_cfg.force)

    def set_gripper_closed_norm(self, closed_norm: float) -> None:
        if not self._gripper_cfg.enable:
            return
        if self._gripper is None:
            raise RuntimeError("Gripper command requested but RobotiqGripper is not available.")
        closed_norm = float(np.clip(closed_norm, 0.0, 1.0))
        lo = int(self._gripper_cfg.open_position)
        hi = int(self._gripper_cfg.closed_position)
        target = int(round(lo + closed_norm * (hi - lo)))
        self._gripper.move(target, speed=self._gripper_cfg.speed, force=self._gripper_cfg.force)

    def enable_freedrive(self) -> None:
        self._rtde.freedriveMode()

    def disable_freedrive(self) -> None:
        self._rtde.endFreedriveMode()

    def stop_motion(self) -> None:
        try:
            self._rtde.servoStop()
        except Exception:
            pass
        try:
            self._rtde.speedStop()
        except Exception:
            pass
