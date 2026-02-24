from __future__ import annotations

from dataclasses import dataclass
import json
import pickle
from pathlib import Path
import threading
import time
from typing import List, Optional

import numpy as np

from ip.deployment.control.ur_rtde_control import URRTDEControl
from ip.deployment.state.ur_rtde_state import URRTDEState

try:
    import serial
except Exception as exc:  # pragma: no cover - dependency may be missing in some environments
    serial = None
    _SERIAL_IMPORT_ERROR = exc
else:
    _SERIAL_IMPORT_ERROR = None


LIGHTNING_OFFSET_RAD = [
    -1.0215797424316406,
    -4.490872740745544,
    -1.4827108010649681,
    -0.588315486907959,
    -0.5356001891195774,
    2.0629922747612,
    0.0,
]

THUNDER_OFFSET_RAD = [
    0.8322513103485107,
    1.3889789581298828,
    1.4154774993658066,
    -2.7204548865556717,
    -2.634120313450694,
    -2.2259570360183716,
    0.0,
]


@dataclass(frozen=True)
class SparkProfile:
    name: str
    joint_offset_rad: List[float]
    invert: List[int]
    grip_raw_min: float
    grip_raw_max: float
    offsets_filename: str


SPARK_PROFILES = {
    "lightning": SparkProfile(
        name="lightning",
        joint_offset_rad=list(LIGHTNING_OFFSET_RAD),
        invert=[-1, -1, 1, -1, -1, -1, -1],
        grip_raw_min=-4.8,
        grip_raw_max=-2.3,
        offsets_filename="offsets_lightning.pickle",
    ),
    "thunder": SparkProfile(
        name="thunder",
        joint_offset_rad=list(THUNDER_OFFSET_RAD),
        invert=[-1, -1, 1, -1, -1, -1, 1],
        grip_raw_min=-2.71,
        grip_raw_max=-1.26,
        offsets_filename="offsets_thunder.pickle",
    ),
}


def _discover_offsets_pickle(filename: str) -> str:
    module_path = Path(__file__).resolve()
    candidates: list[Path] = []
    for parent in module_path.parents:
        candidates.append(parent / "SPARK-Remote-data_collection" / "TeleopSoftware" / "Spark" / filename)
    candidates.append(Path.cwd() / "SPARK-Remote-data_collection" / "TeleopSoftware" / "Spark" / filename)
    candidates.append(Path.cwd() / ".." / "SPARK-Remote-data_collection" / "TeleopSoftware" / "Spark" / filename)
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            return str(resolved)
    return ""


def _map_value(
    x: float,
    in_min: float,
    in_max: float,
    out_min: float = 0.0,
    out_max: float = 1.0,
) -> float:
    if in_max == in_min:
        return out_min
    return out_min + (x - in_min) * (out_max - out_min) / (in_max - in_min)


@dataclass
class SparkPacket:
    timestamp: float
    device_id: str
    raw_values: List[int]
    status: List[bool]
    enable_switch: bool


@dataclass
class SparkSample:
    timestamp: float
    device_id: str
    angles_rad: List[float]
    raw_values: List[int]
    status: List[bool]
    enable_switch: bool


@dataclass
class SparkMonitorState:
    command_joints: Optional[np.ndarray]
    spark_enable: Optional[bool]
    stale: bool
    last_error: Optional[str]


@dataclass
class SparkDeviceConfig:
    arm: str
    serial_device: str
    offsets_pickle: str = ""
    invert: List[int] = None
    baudrate: int = 921600
    timeout_s: float = 0.2
    encoder_modulus: int = 16384

    def __post_init__(self) -> None:
        if self.invert is None:
            self.invert = [-1, -1, 1, -1, -1, -1, -1]
        if len(self.invert) != 7:
            raise ValueError(f"invert must have length 7, got {len(self.invert)}")


class SparkAngleUnwrapper:
    def __init__(self, offsets_raw: List[int], invert: List[int], modulus: int):
        if len(offsets_raw) != 7:
            raise ValueError(f"offsets_raw must have length 7, got {len(offsets_raw)}")
        if len(invert) != 7:
            raise ValueError(f"invert must have length 7, got {len(invert)}")
        self._prev = np.asarray(offsets_raw, dtype=np.int64)
        self._invert = np.asarray(invert, dtype=np.float64)
        self._modulus = int(modulus)
        self._angles = np.zeros(7, dtype=np.float64)

    def update(self, raw_values: List[int]) -> np.ndarray:
        cur = np.asarray(raw_values, dtype=np.int64)
        if cur.shape != (7,):
            raise ValueError(f"raw_values must have shape (7,), got {cur.shape}")
        dist = cur - self._prev
        half = self._modulus / 2
        dist = np.where(dist > half, dist - self._modulus, dist)
        dist = np.where(dist < -half, dist + self._modulus, dist)
        self._angles += self._invert * (dist.astype(np.float64) / float(self._modulus)) * (2.0 * np.pi)
        self._prev = cur
        return self._angles.copy()


class SparkSerialReader:
    def __init__(self, serial_device: str, baudrate: int, timeout_s: float):
        if serial is None:
            raise ImportError(
                "pyserial is required for Spark serial input. Install with `pip install pyserial`."
            ) from _SERIAL_IMPORT_ERROR
        self._serial_device = serial_device
        self._baudrate = baudrate
        self._timeout_s = timeout_s
        self._con = None

    def connect(self) -> None:
        if self._con is not None:
            return
        self._con = serial.Serial(self._serial_device, self._baudrate, timeout=self._timeout_s)
        time.sleep(0.5)
        self._con.reset_input_buffer()
        self._con.reset_output_buffer()
        self._con.read_until(b"\x00")

    def close(self) -> None:
        if self._con is not None:
            try:
                self._con.close()
            finally:
                self._con = None

    def read_packet(self) -> Optional[SparkPacket]:
        if self._con is None:
            raise RuntimeError("SparkSerialReader is not connected.")
        payload = self._con.read_until(b"\x00")
        if not payload:
            return None
        data = payload[:-1] if payload.endswith(b"\x00") else payload
        if not data:
            return None
        parsed = json.loads(data.decode("utf-8"))
        raw_values = parsed.get("values", [])
        if len(raw_values) < 7:
            raise ValueError(f"Spark packet has {len(raw_values)} values; expected at least 7.")
        raw = [int(v) for v in raw_values[:7]]
        status = [bool(v) for v in parsed.get("status", [True] * 7)[:7]]
        return SparkPacket(
            timestamp=time.time(),
            device_id=str(parsed.get("ID", "")),
            raw_values=raw,
            status=status,
            enable_switch=bool(parsed.get("enable_switch", False)),
        )


def _load_offsets_pickle(path: str) -> tuple[List[int], Optional[List[int]]]:
    payload = pickle.load(Path(path).open("rb"))
    if isinstance(payload, tuple) and len(payload) >= 2:
        offsets_raw = [int(v) for v in payload[0][:7]]
        invert = [int(v) for v in payload[1][:7]]
        return offsets_raw, invert
    offsets_raw = [int(v) for v in payload[:7]]
    return offsets_raw, None


class SparkDevice:
    def __init__(self, config: SparkDeviceConfig):
        self.config = config
        self._reader = SparkSerialReader(
            serial_device=config.serial_device,
            baudrate=config.baudrate,
            timeout_s=config.timeout_s,
        )
        self._stop = threading.Event()
        self._thread = None
        self._lock = threading.Lock()
        self._latest = None
        self._unwrap = None
        self._last_error = None

    def _build_unwrapper(self, first_packet: SparkPacket) -> SparkAngleUnwrapper:
        offsets_raw = first_packet.raw_values
        invert = list(self.config.invert)
        if self.config.offsets_pickle:
            pickle_path = Path(self.config.offsets_pickle)
            if not pickle_path.exists():
                raise FileNotFoundError(
                    f"Offsets pickle does not exist for arm '{self.config.arm}': {pickle_path}"
                )
            file_offsets, file_invert = _load_offsets_pickle(str(pickle_path))
            offsets_raw = file_offsets
            if file_invert is not None:
                invert = file_invert
        return SparkAngleUnwrapper(
            offsets_raw=offsets_raw,
            invert=invert,
            modulus=self.config.encoder_modulus,
        )

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._reader.connect()
                packet = self._reader.read_packet()
                if packet is None:
                    continue
                if self._unwrap is None:
                    self._unwrap = self._build_unwrapper(packet)
                angles = self._unwrap.update(packet.raw_values).tolist()
                sample = SparkSample(
                    timestamp=packet.timestamp,
                    device_id=packet.device_id,
                    angles_rad=angles,
                    raw_values=list(packet.raw_values),
                    status=list(packet.status),
                    enable_switch=packet.enable_switch,
                )
                with self._lock:
                    self._latest = sample
                    self._last_error = None
            except Exception as exc:
                with self._lock:
                    self._last_error = str(exc)
                try:
                    self._reader.close()
                except Exception:
                    pass
                time.sleep(0.2)

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._reader.close()

    def get_latest(self) -> Optional[SparkSample]:
        with self._lock:
            return self._latest

    def get_last_error(self) -> Optional[str]:
        with self._lock:
            return self._last_error


class SparkDemoInput:
    def __init__(
        self,
        state: URRTDEState,
        control: URRTDEControl,
        serial_device: str,
        profile_name: str = "lightning",
        offsets_pickle: Optional[str] = None,
        command_rate_hz: float = 200.0,
        stale_timeout_s: float = 0.25,
    ):
        profile_key = str(profile_name).lower()
        if profile_key not in SPARK_PROFILES:
            raise ValueError(
                f"Unsupported Spark profile {profile_name!r}. "
                f"Expected one of: {sorted(SPARK_PROFILES.keys())}"
            )
        self.profile = SPARK_PROFILES[profile_key]
        discovered = _discover_offsets_pickle(self.profile.offsets_filename)
        selected_offsets_pickle = offsets_pickle if offsets_pickle is not None else discovered
        self._state = state
        self._control = control
        self._rate_hz = float(command_rate_hz)
        self._stale_timeout_s = float(stale_timeout_s)
        self._device = SparkDevice(
            SparkDeviceConfig(
                arm=self.profile.name,
                serial_device=serial_device,
                offsets_pickle=selected_offsets_pickle or "",
                invert=list(self.profile.invert),
            )
        )
        self._stop = threading.Event()
        self._thread = None
        self._lock = threading.Lock()
        self._last_open_command = None
        self._last_error = None
        self._was_commanding = False
        self._base_wrap_adjust_rad = 0.0
        self._wrap_initialized = False
        self._last_command_joints = None

    def _run(self) -> None:
        period = 1.0 / max(self._rate_hz, 1e-6)
        offset = np.asarray(self.profile.joint_offset_rad, dtype=np.float64).reshape(7)
        while not self._stop.is_set():
            t0 = time.time()
            try:
                sample = self._device.get_latest()
                stale = True
                if sample is not None:
                    stale = (t0 - float(sample.timestamp)) > self._stale_timeout_s
                can_command = sample is not None and (not stale) and bool(sample.enable_switch)

                if not can_command and self._was_commanding:
                    self._control.stop_motion()
                    self._was_commanding = False

                if can_command:
                    spark_angles = np.asarray(sample.angles_rad, dtype=np.float64).reshape(7)
                    if not self._wrap_initialized:
                        actual_q = self._state.get_actual_q()
                        dq0 = float(spark_angles[0] - actual_q[0] + offset[0])
                        if dq0 > np.pi:
                            self._base_wrap_adjust_rad = -2.0 * np.pi
                        elif dq0 < -np.pi:
                            self._base_wrap_adjust_rad = 2.0 * np.pi
                        self._wrap_initialized = True

                    command_joints = spark_angles[:6] + offset[:6]
                    command_joints[0] += self._base_wrap_adjust_rad
                    ok = self._control.execute_joint_positions(command_joints.tolist())
                    if not ok:
                        raise RuntimeError("Spark servoJ command failed")

                    grip_raw = float(spark_angles[6] + offset[6])
                    grip_closed = float(
                        np.round(
                            np.clip(
                                _map_value(
                                    grip_raw,
                                    in_min=self.profile.grip_raw_min,
                                    in_max=self.profile.grip_raw_max,
                                    out_min=0.0,
                                    out_max=1.0,
                                ),
                                0.0,
                                1.0,
                            )
                            * 10.0
                        )
                        / 10.0
                    )
                    self._control.set_gripper_closed_norm(grip_closed)
                    with self._lock:
                        self._last_open_command = float(1.0 - grip_closed)
                        self._last_command_joints = np.asarray(command_joints, dtype=np.float64).copy()
                    self._was_commanding = True

                with self._lock:
                    self._last_error = self._device.get_last_error()
            except Exception as exc:
                with self._lock:
                    self._last_error = str(exc)
                try:
                    self._control.stop_motion()
                except Exception:
                    pass
                self._was_commanding = False

            elapsed = time.time() - t0
            if elapsed < period:
                time.sleep(period - elapsed)

    def start(self) -> None:
        self._device.start()
        deadline = time.time() + 3.0
        while time.time() < deadline:
            sample = self._device.get_latest()
            if sample is not None:
                break
            err = self._device.get_last_error()
            if err:
                self._device.stop()
                raise RuntimeError(f"Failed to initialize Spark input: {err}")
            time.sleep(0.05)
        else:
            self._device.stop()
            raise RuntimeError("Timed out waiting for initial Spark packet.")
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._device.stop()
        try:
            self._control.stop_motion()
        except Exception:
            pass

    def get_last_open_command(self) -> Optional[float]:
        with self._lock:
            return self._last_open_command

    def get_last_error(self) -> Optional[str]:
        with self._lock:
            return self._last_error

    def get_monitor_state(self) -> SparkMonitorState:
        sample = self._device.get_latest()
        with self._lock:
            cmd = None if self._last_command_joints is None else self._last_command_joints.copy()
            last_error = self._last_error
        stale = True
        spark_enable = None
        if sample is not None:
            spark_enable = bool(sample.enable_switch)
            stale = (time.time() - float(sample.timestamp)) > self._stale_timeout_s
        return SparkMonitorState(
            command_joints=cmd,
            spark_enable=spark_enable,
            stale=stale,
            last_error=last_error,
        )
