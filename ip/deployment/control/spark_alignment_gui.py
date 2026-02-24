from __future__ import annotations

import tkinter as tk
import time
from typing import Callable, Optional
from math import cos
from math import sin

import numpy as np

from ip.deployment.control.spark_input import SparkDemoInput
from ip.deployment.state.ur_rtde_state import URRTDEState


# UR5e DH parameters (same model used in legacy Spark monitor path).
d1 = 0.163
a2 = -0.425
a3 = -0.392
d4 = 0.127
d5 = 0.1
d6 = 0.1


def _forward_xyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(6)
    s = [sin(q[0]), sin(q[1]), sin(q[2]), sin(q[3]), sin(q[4]), sin(q[5])]
    c = [cos(q[0]), cos(q[1]), cos(q[2]), cos(q[3]), cos(q[4]), cos(q[5])]
    q23 = q[1] + q[2]
    q234 = q[1] + q[2] + q[3]
    s23 = sin(q23)
    c23 = cos(q23)
    s234 = sin(q234)
    c234 = cos(q234)

    x = (
        d6 * c234 * c[0] * s[4]
        - a3 * c23 * c[0]
        - a2 * c[0] * c[1]
        - d6 * c[4] * s[0]
        - d5 * s234 * c[0]
        - d4 * s[0]
    )
    y = (
        d6 * (c[0] * c[4] + c234 * s[0] * s[4])
        + d4 * c[0]
        - a3 * c23 * s[0]
        - a2 * c[1] * s[0]
        - d5 * s234 * s[0]
    )
    z = d1 + a3 * s23 + a2 * s[1] - d5 * (c23 * c[3] - s23 * s[3]) - d6 * s[4] * (
        c23 * s[3] + s23 * c[3]
    )
    return np.asarray([x, y, z], dtype=np.float64)


def _compute_xyz_error(command_joints: np.ndarray, actual_joints: np.ndarray, profile_name: str) -> tuple[float, float, float]:
    spark_xyz = _forward_xyz(command_joints)
    ur_xyz = _forward_xyz(actual_joints)
    arm = profile_name.lower()
    if arm == "thunder":
        x = -spark_xyz[1] + ur_xyz[1]
        y = -spark_xyz[0] + ur_xyz[0]
        z = spark_xyz[2] - ur_xyz[2]
    else:
        x = -((spark_xyz[1] - ur_xyz[1]))
        y = spark_xyz[0] - ur_xyz[0]
        z = -(spark_xyz[2] - ur_xyz[2])
    return float(x), float(y), float(z)


class SparkAlignmentGui:
    def __init__(
        self,
        spark_input: SparkDemoInput,
        state: URRTDEState,
        profile_name: str,
        xy_bound_m: float = 0.12,
        z_bound_m: float = 0.12,
        refresh_hz: float = 30.0,
    ):
        self._spark_input = spark_input
        self._state = state
        self._profile_name = profile_name
        self._xy_bound_m = float(xy_bound_m)
        self._z_bound_m = float(z_bound_m)
        self._refresh_hz = float(refresh_hz)
        self._start_requested = False
        self._stop_requested = False
        self._save_requested = False
        self._closed = False
        self._last_pump_t = 0.0

        self._root = tk.Tk()
        self._root.title("Spark Alignment Check")
        self._root.geometry("560x560")
        self._root.protocol("WM_DELETE_WINDOW", self._on_cancel)

        self._status = tk.Label(self._root, text="status: initializing", fg="black")
        self._status.pack(anchor="w", padx=12, pady=6)

        self._plot = tk.Canvas(self._root, width=380, height=380, bg="white")
        self._plot.pack(side=tk.LEFT, padx=12, pady=8)
        cx, cy = 190, 190
        radius = int(self._xy_bound_m * 400.0)
        self._plot.create_oval(cx - radius, cy - radius, cx + radius, cy + radius, outline="black", width=2)
        self._point = self._plot.create_oval(cx - 8, cy - 8, cx + 8, cy + 8, fill="blue")

        self._z_canvas = tk.Canvas(self._root, width=80, height=380, bg="white")
        self._z_canvas.pack(side=tk.LEFT, padx=8, pady=8)
        z_bound_px = int(self._z_bound_m * 350.0)
        self._z_canvas.create_rectangle(18, 190 - z_bound_px, 62, 190 + z_bound_px, outline="black", width=2)
        self._z_rect = self._z_canvas.create_rectangle(20, 180, 60, 200, fill="blue")

        self._info = tk.Label(self._root, text="dx=nan dy=nan dz=nan", justify=tk.LEFT)
        self._info.pack(anchor="w", padx=12, pady=6)

        btn_frame = tk.Frame(self._root)
        btn_frame.pack(fill=tk.X, padx=12, pady=8)
        self._start_btn = tk.Button(btn_frame, text="Start Recording", width=14, command=self._on_start)
        self._start_btn.pack(side=tk.LEFT)
        self._stop_btn = tk.Button(btn_frame, text="Stop + Save", width=12, command=self._on_stop, state=tk.DISABLED)
        self._stop_btn.pack(side=tk.LEFT, padx=8)
        self._cancel_btn = tk.Button(btn_frame, text="Cancel", width=12, command=self._on_cancel)
        self._cancel_btn.pack(side=tk.LEFT)

    def _on_start(self) -> None:
        if self._closed or self._start_requested:
            return
        self._start_requested = True
        self._start_btn.config(state=tk.DISABLED)
        self._stop_btn.config(state=tk.NORMAL)

    def _on_stop(self) -> None:
        if self._closed:
            return
        if not self._start_requested:
            return
        self._save_requested = True
        self._stop_requested = True
        self.close()

    def _on_cancel(self) -> None:
        if self._closed:
            return
        self._save_requested = False
        self._stop_requested = True
        self.close()

    def _refresh(self) -> None:
        cx, cy = 190, 190
        try:
            monitor = self._spark_input.get_monitor_state()
            if monitor.command_joints is None:
                raise RuntimeError("No Spark command yet. Move Spark and enable switch.")
            actual_q = self._state.get_actual_q()
            dx, dy, dz = _compute_xyz_error(monitor.command_joints, actual_q, self._profile_name)

            x_px = int(round(cx + dx * 400.0))
            y_px = int(round(cy - dy * 400.0))
            x_px = max(0, min(380, x_px))
            y_px = max(0, min(380, y_px))
            self._plot.moveto(self._point, x_px - 8, y_px - 8)

            z_px = int(round(cy - dz * 350.0))
            z_px = max(0, min(380, z_px))
            self._z_canvas.coords(self._z_rect, 20, z_px - 10, 60, z_px + 10)

            in_bounds = (
                abs(dx) <= self._xy_bound_m
                and abs(dy) <= self._xy_bound_m
                and abs(dz) <= self._z_bound_m
            )
            color = "#3cb371" if in_bounds else "#ff8c00"
            self._plot.itemconfig(self._point, fill=color)
            self._z_canvas.itemconfig(self._z_rect, fill=color)
            self._status.config(
                text=(
                    f"status: {'recording' if self._start_requested else 'ready'} | "
                    f"spark_enable={monitor.spark_enable} | "
                    f"{'stale' if monitor.stale else 'live'}"
                ),
                fg="black" if in_bounds else "#cc5500",
            )
            self._info.config(text=f"dx={dx:+.3f} dy={dy:+.3f} dz={dz:+.3f} m")
        except Exception as exc:
            self._plot.itemconfig(self._point, fill="blue")
            self._z_canvas.itemconfig(self._z_rect, fill="blue")
            self._status.config(
                text=f"status: {'recording' if self._start_requested else 'ready'} | waiting ({exc})",
                fg="#7a7a7a",
            )
            self._info.config(text="dx=nan dy=nan dz=nan")

    def pump(self) -> None:
        if self._closed:
            return
        now = time.time()
        period = 1.0 / max(self._refresh_hz, 1e-3)
        if (now - self._last_pump_t) >= period:
            self._refresh()
            self._last_pump_t = now
        try:
            self._root.update_idletasks()
            self._root.update()
        except tk.TclError:
            self._closed = True
            self._stop_requested = True

    def wait_for_start(self, external_stop: Optional[Callable[[], bool]] = None) -> bool:
        while not self._closed:
            if external_stop is not None and external_stop():
                return False
            self.pump()
            if self._start_requested:
                return True
            if self._stop_requested:
                return False
            time.sleep(0.01)
        return False

    def should_stop(self) -> bool:
        return bool(self._stop_requested or self._closed)

    def should_save(self) -> bool:
        return bool(self._save_requested)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._root.destroy()
        except Exception:
            pass
