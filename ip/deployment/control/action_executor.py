from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass
class SafetyLimits:
    max_translation: float = 0.01
    max_rotation: float = np.deg2rad(3.0)


class ActionExecutor:
    TARGET_TRANS_TOL_M = 5e-4
    TARGET_ROT_TOL_RAD = np.deg2rad(0.2)
    MAX_SUBSTEPS_PER_ACTION = 200

    def __init__(self, control, state, safety: SafetyLimits = None, debug_gripper: bool = False):
        self.control = control
        self.state = state
        self.safety = safety or SafetyLimits()
        self._debug_gripper = debug_gripper

    def execute_actions(
        self,
        actions: np.ndarray,
        grips: np.ndarray,
        T_w_e_initial: np.ndarray,
        horizon: int = 8,
    ) -> Tuple[bool, int, str]:
        # Each action is relative to the pose at inference time (T_w_e_initial).
        T_w_e_base = T_w_e_initial.copy()
        steps = min(horizon, len(actions))
        steps_executed = 0
        last_grip_state = None

        for j in range(steps):
            T_w_e_target = T_w_e_base @ actions[j]
            try:
                q_near = self.state.get_actual_q()
                ok, reason = self.control.validate_target_pose(T_w_e_target, q_near)
            except Exception as exc:
                return False, steps_executed, f"Kinematic precheck failed: {exc}"
            if not ok:
                return False, steps_executed, reason

            grip_cmd = (grips[j] + 1) / 2
            desired = 1 if grip_cmd >= 0.5 else 0
            if self._debug_gripper and (last_grip_state is None or desired != last_grip_state):
                state_label = "open" if desired == 1 else "close"
                print(f"[gripper] step {j}: cmd={grip_cmd:.3f} -> {state_label}")
            last_grip_state = desired
            self.control.execute_gripper(grip_cmd)
            substeps = 0
            while True:
                T_w_e_current = self.state.get_T_w_e()
                if self._is_target_reached(T_w_e_current, T_w_e_target):
                    break
                if substeps >= self.MAX_SUBSTEPS_PER_ACTION:
                    return (
                        False,
                        steps_executed,
                        f"Target not reached within {self.MAX_SUBSTEPS_PER_ACTION} bounded substeps",
                    )

                T_w_e_next, _ = self._bounded_step(T_w_e_current, T_w_e_target)

                ok, reason = self._check_safety(T_w_e_current, T_w_e_next)
                if not ok:
                    return False, steps_executed, reason

                if not self.control.execute_pose(T_w_e_next):
                    return False, steps_executed, "Motion execution failed"

                steps_executed += 1
                substeps += 1

        return True, steps_executed, "Success"

    def _is_target_reached(self, T_current: np.ndarray, T_target: np.ndarray) -> bool:
        T_err = np.linalg.inv(T_current) @ T_target
        trans = np.linalg.norm(T_err[:3, 3])
        rot = Rotation.from_matrix(T_err[:3, :3]).magnitude()
        return trans <= self.TARGET_TRANS_TOL_M and rot <= self.TARGET_ROT_TOL_RAD

    def _check_safety(self, T_prev: np.ndarray, T_next: np.ndarray) -> Tuple[bool, str]:
        trans = np.linalg.norm(T_next[:3, 3] - T_prev[:3, 3])
        rot = Rotation.from_matrix(T_prev[:3, :3].T @ T_next[:3, :3]).magnitude()
        if trans > self.safety.max_translation + 1e-9:
            return False, "Translation exceeds per-step limit"
        if rot > self.safety.max_rotation + 1e-9:
            return False, "Rotation exceeds per-step limit"
        return True, ""

    def _bounded_step(self, T_prev: np.ndarray, T_target: np.ndarray) -> Tuple[np.ndarray, bool]:
        T_err = np.linalg.inv(T_prev) @ T_target
        trans_vec = T_err[:3, 3]
        rot_vec = Rotation.from_matrix(T_err[:3, :3]).as_rotvec()
        trans = np.linalg.norm(trans_vec)
        rot = np.linalg.norm(rot_vec)

        if trans <= self.safety.max_translation and rot <= self.safety.max_rotation:
            return T_target, True

        scale = 1.0
        if trans > 0 and self.safety.max_translation > 0:
            scale = min(scale, self.safety.max_translation / trans)
        if rot > 0 and self.safety.max_rotation > 0:
            scale = min(scale, self.safety.max_rotation / rot)

        T_step = np.eye(4, dtype=np.float64)
        T_step[:3, :3] = Rotation.from_rotvec(rot_vec * scale).as_matrix()
        T_step[:3, 3] = trans_vec * scale
        return T_prev @ T_step, False
