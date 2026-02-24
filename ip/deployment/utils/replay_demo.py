#!/usr/bin/env python3
"""
Replay a recorded demo trajectory on hardware.

Robot RTDE TCP is assumed to be flange.
Policy frame is flange translated by a fixed flange->policy-origin offset.
"""
import argparse
import pickle
from pathlib import Path

import numpy as np

from ip.deployment.config import GripperConfig, RTDEControlConfig
from ip.deployment.control.action_executor import ActionExecutor, SafetyLimits
from ip.deployment.control.ur_rtde_control import URRTDEControl
from ip.deployment.state.ur_rtde_state import URRTDEState
from ip.deployment.control.robotiq_gripper import RobotiqGripper

REPLAY_MOVE_SPEED_M_S = 0.25
REPLAY_MOVE_ACCEL_M_S2 = 1.2


def _prompt(msg: str) -> bool:
    resp = input(f"{msg} [y/N] ").strip().lower()
    return resp in {"y", "yes"}


def _load_demo(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def _build_frame_spec(flange_to_policy_origin_m: np.ndarray) -> dict:
    return {
        "robot_tcp_frame": "flange",
        "flange_to_policy_origin_m": [
            float(x) for x in np.asarray(flange_to_policy_origin_m, dtype=np.float64).reshape(3)
        ],
    }


def _offset_from_demo_frame_spec(spec: dict, demo_path: str) -> np.ndarray:
    if "flange_to_policy_origin_m" not in spec:
        raise ValueError(
            f"Demo {demo_path} frame_spec is missing required key 'flange_to_policy_origin_m'."
        )
    offset = np.asarray(spec["flange_to_policy_origin_m"], dtype=np.float64)
    if offset.shape != (3,):
        raise ValueError(
            f"Demo {demo_path} has invalid flange_to_policy_origin_m shape {offset.shape}; expected (3,)."
        )
    return offset


def _validate_demo_frame_spec(demo: dict, demo_path: str, expected_spec: dict) -> None:
    spec = demo.get("frame_spec")
    if spec is None:
        raise ValueError(
            f"Demo {demo_path} is missing required frame_spec metadata."
        )

    robot_tcp = str(spec.get("robot_tcp_frame", "")).lower()
    if robot_tcp != "flange":
        raise ValueError(
            f"Demo {demo_path} has unsupported robot_tcp_frame={robot_tcp!r}. Expected 'flange'."
        )

    demo_offset = _offset_from_demo_frame_spec(spec, demo_path)
    expected_offset = np.asarray(expected_spec["flange_to_policy_origin_m"], dtype=np.float64)
    if not np.allclose(demo_offset, expected_offset, atol=1e-6):
        raise ValueError(
            f"Frame mismatch for demo {demo_path}: demo flange_to_policy_origin_m="
            f"{demo_offset.tolist()} != current replay {expected_offset.tolist()}."
        )


def _indices(n: int, start: int, end: int, stride: int):
    start = max(0, start)
    end = n - 1 if end < 0 else min(end, n - 1)
    return list(range(start, end + 1, max(1, stride)))


def main():
    parser = argparse.ArgumentParser(description="Replay a demo trajectory on hardware.")
    parser.add_argument("--demo", required=True, help="Path to demo .pkl")
    parser.add_argument("--robot-ip", required=True, help="Robot IP address")
    parser.add_argument("--start", type=int, default=0, help="Start frame index")
    parser.add_argument("--end", type=int, default=-1, help="End frame index (inclusive, -1 for last)")
    parser.add_argument("--stride", type=int, default=1, help="Play every Nth frame")
    parser.add_argument("--go-start", action="store_true", help="Move to first pose before replay")
    parser.add_argument("--no-confirm", action="store_true", help="Skip confirmation prompts")
    parser.add_argument("--dry-run", action="store_true", help="Print actions but do not move")
    parser.add_argument(
        "--flange-to-policy-origin-m",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Policy origin offset from flange in meters (applied in code to RTDE flange pose).",
    )
    parser.add_argument("--use-gripper", action="store_true", help="Replay gripper commands from demo")
    args = parser.parse_args()

    demo_path = Path(args.demo)
    demo = _load_demo(demo_path)
    T_w_es = demo.get("T_w_es", [])
    grips = demo.get("grips", [])
    if not T_w_es:
        raise RuntimeError("Demo has no T_w_es.")
    if args.use_gripper and not grips:
        raise RuntimeError("Demo has no grips; re-collect with gripper states or omit --use-gripper.")
    idxs = _indices(len(T_w_es), args.start, args.end, args.stride)
    if not idxs:
        raise RuntimeError("No frames in selected range.")

    tcp_offset_in_code = True
    tcp_offset = np.array([0.0, 0.0, 0.088], dtype=np.float64)
    if args.flange_to_policy_origin_m is not None:
        tcp_offset = np.array(args.flange_to_policy_origin_m, dtype=np.float64)
    frame_spec = _build_frame_spec(flange_to_policy_origin_m=tcp_offset)
    _validate_demo_frame_spec(demo, str(demo_path), frame_spec)

    gripper = None
    if args.use_gripper:
        gripper = RobotiqGripper(
            host=args.robot_ip,
        )
        gripper.connect()
        gripper.activate()

    rtde_cfg = RTDEControlConfig(
        control_mode="moveL",
        move_speed=REPLAY_MOVE_SPEED_M_S,
        move_acceleration=REPLAY_MOVE_ACCEL_M_S2,
    )
    rtde_control = URRTDEControl.connect(args.robot_ip, rtde_cfg)
    rtde_receive_iface = URRTDEState.connect(args.robot_ip)
    control = URRTDEControl(
        rtde_control,
        rtde_cfg,
        gripper=gripper,
        gripper_config=GripperConfig(
            enable=args.use_gripper,
            host=args.robot_ip,
        ),
        tcp_offset_in_code=tcp_offset_in_code,
        tcp_offset_m=tcp_offset,
    )
    state = URRTDEState(
        rtde_receive_iface,
        gripper=gripper,
        tcp_offset_in_code=tcp_offset_in_code,
        tcp_offset_m=tcp_offset,
    )
    safety = SafetyLimits()
    executor = ActionExecutor(control, state, safety=safety, debug_gripper=False)

    print("ROBOT RTDE TCP FRAME = FLANGE (fixed)")
    print(f"POLICY ORIGIN OFFSET FROM FLANGE (m) = {tcp_offset.tolist()}")
    print(f"Replaying {len(idxs)} frames (stride={args.stride}).")

    if args.go_start:
        first_idx = idxs[0]
        T_start = np.asarray(T_w_es[first_idx], dtype=np.float64)
        if not args.no_confirm:
            if not _prompt(f"Move to start frame {first_idx}?"):
                return
        if args.dry_run:
            print("Dry run: would move to start pose.")
        else:
            control.execute_pose(T_start)

    if not args.no_confirm:
        if not _prompt("Begin replay?"):
            return

    for i, idx in enumerate(idxs):
        T_target = np.asarray(T_w_es[idx], dtype=np.float64)
        if T_target.shape != (4, 4):
            raise RuntimeError(f"Invalid T_w_e shape at frame {idx}: {T_target.shape} (expected (4, 4))")
        grip_val = None
        if args.use_gripper and idx < len(grips):
            grip_val = 1.0 if grips[idx] >= 0.5 else 0.0
        if args.dry_run:
            print(f"[{i+1}/{len(idxs)}] frame {idx} (dry run)")
            continue

        T_current = state.get_T_w_e()
        T_rel = np.linalg.inv(T_current) @ T_target
        actions = np.expand_dims(T_rel, axis=0)
        if grip_val is None:
            grips_rel = np.array([1.0], dtype=np.float64)
        else:
            grips_rel = np.array([1.0 if grip_val >= 0.5 else -1.0], dtype=np.float64)
        ok, steps, reason = executor.execute_actions(actions, grips_rel, T_current, horizon=1)
        if not ok:
            print(f"Stopped at frame {idx}: {reason}")
            break


if __name__ == "__main__":
    main()
