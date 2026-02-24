# Instant Policy Deployment: Step-by-Step Setup Guide

This guide walks through deploying Instant Policy on a **brand new system** from scratch. It covers hardware setup, software installation, calibration, and first run.

For SPARK-as-input demo collection (shared deployment pipeline), see:
- `ip/deployment/docs/SPARK_TELEOP_GUIDE.md`

---

## Prerequisites

### Hardware Required
- **Robot**: UR5e with Robotiq 2F-85 gripper
- **Cameras**: Intel RealSense L515 or D455 (1-2 cameras)
  Note: 2 L515 cameras will interfere with each other.
- **Compute**: Linux workstation with NVIDIA GPU (CUDA required for XMem++)
- **Network**: Ethernet connection to robot (same subnet)

### Software Required
- Ubuntu 22.04 or 24.04
- NVIDIA drivers + CUDA 11.8+
- Python 3.10+
- Conda or Mamba

---

## Step 1: Network Setup

### 1.1 Configure Robot IP
On the UR teach pendant, go to **Settings → Network**:
- IP address: `10.33.55.90`
- Subnet mask: `255.255.255.0`
- Default gateway: `10.33.55.1`

### 1.2 Configure Workstation IP
```bash
# Set static IP on same subnet
sudo ip addr add 10.33.55.100/24 dev enp1s0f0
sudo ip link set enp1s0f0 up
```

### 1.3 Verify Connectivity
```bash
ping 10.33.55.90
```

---

## Step 2: Install Dependencies

### 2.1 Create Conda Environment from environment.yml
The Instant Policy repo includes a complete `environment.yml` with all dependencies:

```bash
cd /path/to/instant_policy
conda env create -f environment.yml
conda activate ip_env
```

This installs:
- Python 3.10
- PyTorch 2.2.0 with CUDA 11.8
- PyTorch Geometric (pyg) + cluster/scatter
- PyTorch Lightning
- NumPy, SciPy, Open3D, and more

### 2.2 Install PyG-lib (required for graph operations)
```bash
pip install pyg-lib -f https://data.pyg.org/whl/torch-2.2.0+cu118.html
```

### 2.3 Install `ip` Package + Deployment Extras
```bash
cd /path/to/instant_policy
pip install -e ".[deployment,segmentation]"
```

This installs deployment-layer Python dependencies in one command:
- `ur_rtde`
- `pynput`
- `opencv-python`
- `viser`
- `gdown`
- `segment-anything`

Import policy:
- Deployment modules use direct imports for these dependencies.
- Missing packages now fail immediately at startup (no optional import fallback path).

### 2.4 Install Intel RealSense SDK + Viewer + Python bindings (Conda-local)
This flow builds librealsense inside your conda env and avoids system-wide installs.
Only two steps touch the system:
- Installing build/GUI deps via `apt-get` (required)
- Installing a udev rule to persist USB permissions (recommended for non-root access)

#### 2.4.1 System build + GUI dependencies (safe global install)
```bash
sudo apt-get update
sudo apt-get install -y \
  git cmake build-essential pkg-config \
  libusb-1.0-0-dev libudev-dev libssl-dev \
  libgtk-3-dev libglfw3-dev \
  libgl1-mesa-dev libglu1-mesa-dev
```

#### 2.4.2 Build librealsense (RSUSB) into your conda env (includes viewer + tools + python modules)
```bash
conda activate ip_env

mkdir -p ~/src
cd ~/src
git clone https://github.com/IntelRealSense/librealsense.git
cd librealsense
git checkout v2.54.2

rm -rf build
mkdir build && cd build

cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$CONDA_PREFIX" \
  -DFORCE_RSUSB_BACKEND=ON \
  -DBUILD_EXAMPLES=ON \
  -DBUILD_GRAPHICAL_EXAMPLES=ON \
  -DBUILD_PYTHON_BINDINGS=ON \
  -DPYTHON_EXECUTABLE="$(which python)"

make -j"$(nproc)"
make install
```

#### 2.4.3 If build fails with uint64_t / `<cstdint>` (Ubuntu 24.04+ sometimes)
If you see errors mentioning `uint64_t` not declared in:
`third-party/rsutils/include/rsutils/version.h`, apply:
```bash
cd ~/src/librealsense
sed -i '/^#pragma once/a#include <cstdint>' third-party/rsutils/include/rsutils/version.h

cd build
make -j"$(nproc)"
make install
```

#### 2.4.4 Make `pyrealsense2` importable in the env
`make install` places Python extension modules into `$CONDA_PREFIX/OFF/`. Copy them into your env’s site-packages:
```bash
conda activate ip_env
SITEPKG="$(python -c 'import site; print(site.getsitepackages()[0])')"

cp -av "$CONDA_PREFIX/OFF/pyrealsense2"*.so* "$SITEPKG/"
cp -av "$CONDA_PREFIX/OFF/pybackend2"*.so* "$SITEPKG/" 2>/dev/null || true
cp -av "$CONDA_PREFIX/OFF/pyrsutils"*.so*   "$SITEPKG/" 2>/dev/null || true
```

#### 2.4.5 Ensure the env finds the correct runtime libs
```bash
conda activate ip_env

mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
cat > "$CONDA_PREFIX/etc/conda/activate.d/realsense.sh" <<'EOF'
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
EOF

conda deactivate
conda activate ip_env
```

#### 2.4.6 Verify installation (viewer + CLI + Python)
```bash
conda activate ip_env

# 1) CLI sees devices
rs-enumerate-devices

# 2) Viewer launches (must work)
realsense-viewer

# 3) Python bindings import and detect cameras
python -c "import pyrealsense2 as rs; print('OK', rs.__version__); print('devices', len(rs.context().query_devices()))"
```

#### 2.4.7 USB permissions (shared systems)
If `rs-enumerate-devices` or `realsense-viewer` shows no device detected but the camera is visible in `lsusb`, you likely lack access to the USB node.
After reboot or replug, `/dev/bus/usb/...` is recreated, so permissions can regress without a udev rule.

Recommended (stable, minimal global change):
- Ensure your user is in `plugdev` and `video`
- Install the RealSense udev rule once so permissions persist

Install the rule and reload udev:
```bash
sudo cp ~/src/librealsense/config/99-realsense-libusb.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules
sudo udevadm trigger
```

Then replug the camera and verify:
```bash
conda activate ip_env
rs-enumerate-devices
```

No-global-change alternative (not persistent):
- Use `sudo` (and pass `LD_LIBRARY_PATH` if needed) or ask an admin to apply temporary ACLs

#### 2.4.8 Optional sanity check
Confirm you are using the env-local binaries:
```bash
command -v realsense-viewer
ldd "$(which realsense-viewer)" | grep librealsense
```

---

## Step 3: Download Model Weights

### 3.1 Instant Policy Pre-trained Weights
Use the official download script:

```bash
cd /path/to/instant_policy

# Download pre-trained model
bash ip/scripts/download_weights.sh
```

This downloads from Google Drive to a `weights/` folder containing:
- `config.pkl` - Model configuration
- `model.pt` - Pre-trained weights

Alternatively, download manually from:
https://drive.google.com/drive/folders/1hfyQ0DhZ8sCLrrH7dmE4WLMIibxZGVpI

### 3.2 SAM Weights (for initial segmentation)
```bash
mkdir -p checkpoints/sam
cd checkpoints/sam

# Download SAM ViT-B (default for Instant Policy)
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

### 3.3 XMem Weights (for video object tracking)
```bash
mkdir -p checkpoints/xmem
cd checkpoints/xmem

# Download XMem checkpoint (from official hkchengrex/XMem repo)
wget https://github.com/hkchengrex/XMem/releases/download/v1.0/XMem.pth
```

---

## Step 4: Setup XMem++ (XMem2) Repository

XMem++ (XMem2) must be cloned and added to your environment's Python path.

```bash
cd /path/to/instant_policy

# Clone XMem2 from the correct repo (mbzuai-metaverse, NOT hkchengrex)
git clone https://github.com/mbzuai-metaverse/XMem2.git XMem2-main

# Put the XMem weights in XMem2-main/saves/
mkdir -p XMem2-main/saves
cp checkpoints/xmem/XMem.pth XMem2-main/saves/

# Register XMem2 import path in this conda env (persistent for ip_env)
SITEPKG="$(python -c 'import site; print(site.getsitepackages()[0])')"
echo "/path/to/instant_policy/XMem2-main" > "$SITEPKG/xmem2.pth"

# Verify structure:
# instant_policy/
#   ├── XMem2-main/
#   │   ├── model/
#   │   ├── inference/
#   │   ├── saves/
#   │   │   └── XMem.pth
#   │   └── ...
#   └── ip/
```

Verification:
```bash
python -c "from model.network import XMem; print('XMem2 import OK')"
```

---

## Step 5: Camera Calibration

Calibration utilities are grouped under `ip/deployment/calibration/` and are invoked as modules.

### 5.0 Hardware + Calibration Summary (Current Setup)

**Hardware Configuration**
- Robot gripper: Robotiq 2F-85
- Calibration contact offset from flange (Z): `0.162 m` (162 mm)
  - This is used only to convert flange poses to contact points in calibration tools.
  - Robot RTDE TCP is treated as **always flange**.
  - Deployment uses a fixed runtime offset from flange to policy origin.
  - Runtime default is `--flange-to-policy-origin-m 0 0 0.088` (applied in code).
  - This same 0.162 value is used in calibration utilities (for flange-pose corner touches).
- Camera 1 serial: `f1380660`
- Camera 2 serial: `f1371463`
- ArUco marker:
  - Dictionary: `DICT_6X6_50`
  - Tag ID: `5`
  - Physical size: `0.05 m` (50 mm)

**World Tag Measurements (Robot Base Frame)**
**Convention:** ArUco CCW_center (origin at tag center; TL has +Y / TR has +X).
Measured by touching the ArUco marker corners with the closed gripper (Feature: Base).
Record each corner as a **6D flange pose** from RTDE (`x y z rx ry rz`).

Corner order is:
- Top-Left (TL)
- Top-Right (TR)
- Bottom-Right (BR)
- Bottom-Left (BL)

`ip.deployment.calibration.compute_world_tag` now uses all four corners and computes `T_world_tag` with a best-fit plane/frame.
The script applies `--flange-to-contact-m` (default `0 0 0.162`) to convert flange positions to contact points before fitting.
Compute `world_tag.json` from all four corners (ArUco frame: X = TL->TR, Y = BL->TL, Z = X x Y).
Arm-aware defaults:
- `--arm left` -> `ip/deployment/calibration/outputs/world_tag.json`
- `--arm right` -> `ip/deployment/calibration/outputs/world_tag_right.json`
```bash
python -m ip.deployment.calibration.compute_world_tag \
  --arm left \
  --tl <x_tl> <y_tl> <z_tl> <rx_tl> <ry_tl> <rz_tl> \
  --tr <x_tr> <y_tr> <z_tr> <rx_tr> <ry_tr> <rz_tr> \
  --br <x_br> <y_br> <z_br> <rx_br> <ry_br> <rz_br> \
  --bl <x_bl> <y_bl> <z_bl> <rx_bl> <ry_bl> <rz_bl> \
  --tag-size 0.05
```

Calculated `world_tag.json` (T_world_tag):
```
[[-0.0309, -0.9994,  0.0141, -0.4902]
 [ 0.0021,  0.0141,  0.9999,  0.5286]
 [-0.9995,  0.0309,  0.0016, -0.2800]
 [ 0.0000,  0.0000,  0.0000,  1.0000]]
```

**Camera Extrinsics (World → Camera)**
Command used:
```bash
python -m ip.deployment.calibration.calibrate_realsense_aruco \
  --arm left \
  --serial f1380660 \
  --serial f1371463 \
  --tag-dict DICT_6X6_50 \
  --tag-id 5 \
  --tag-size 0.05 \
  --world-tag-matrix ip/deployment/calibration/outputs/world_tag.json
```

Calibration results (saved to arm-specific default):
- `--arm left` -> `realsense_T_world_camera.json`
- `--arm right` -> `realsense_T_world_camera_right.json`

Camera 1 (`f1380660`)
- Position (approx): X = `-0.44 m`, Y = `1.17 m`, Z = `0.086 m`
- Quality: `30 samples`, `0.21 px` reprojection error (excellent)
```
[[ 0.9952,  0.0067, -0.0978, -0.4402]
 [-0.0729, -0.6154, -0.7849,  1.1759]
 [-0.0654,  0.7882, -0.6119,  0.0865]
 [ 0.0000,  0.0000,  0.0000,  1.0000]]
```

Camera 2 (`f1371463`)
- Position (approx): X = `-0.51 m`, Y = `1.50 m`, Z = `-0.012 m`
- Quality: `30 samples`, `0.05 px` reprojection error (excellent)
```
[[ 0.9978,  0.0556, -0.0352, -0.5160]
 [-0.0283, -0.1209, -0.9923,  1.5065]
 [-0.0595,  0.9911, -0.1190, -0.0123]
 [ 0.0000,  0.0000,  0.0000,  1.0000]]
```

**File Locations**
- Tag calibration: `ip/deployment/calibration/outputs/world_tag.json`
- Camera calibration: `ip/deployment/calibration/outputs/realsense_T_world_camera.json`
- Right-arm variants append `_right` in the same folder.

Calibration sanity check (click pixel -> world point, optional TCP comparison):
```bash
python -m ip.deployment.calibration.validate_click_point \
  --arm left \
  --serial <CAMERA_SERIAL> \
  --robot-ip <UR_IP>
```

**Validation (Click‑to‑World vs TCP)**
Example validation using `ip.deployment.calibration.validate_click_point`:
```
Pixel (347,334) depth 0.7290 m
Camera point: [0.0237, 0.1144, 0.7290] m
World point : [-0.4871, 0.5316, -0.2710] m
TCP pose   : [-0.4901, 0.5291, -0.2786, -1.1912, -1.2087, -1.2095]
Delta (TCP - point): [-0.0030, -0.0024, -0.0077] m
```
This shows ~3 mm XY error and ~8 mm Z error, which is within expected RealSense depth noise.

### 5.1 Get Camera Serial Numbers
```bash
# List connected RealSense cameras
rs-enumerate-devices | grep "Serial Number"
```

---

## Step 6: Gripper Setup

### 6.1 Connect Gripper
The Robotiq 2F-85 connects via the UR tool connector. The deployment uses TCP socket communication on port 63352.

### 6.2 Enable URCap
On the UR teach pendant:
1. Go to **Installation → URCaps**
2. Enable the Robotiq gripper URCap
3. The gripper should now be accessible on the robot's IP at port 63352

### 6.3 Test Gripper Connection
```python
from ip.deployment.control.robotiq_gripper import RobotiqGripper

gripper = RobotiqGripper(host="10.33.55.90", port=63352)
gripper.connect()
gripper.activate()
print("Gripper activated!")
gripper.open()
gripper.close()
gripper.disconnect()
```

---

## Step 7: Robot Setup

### 7.1 Test RTDE Connection
```python
import rtde_receive
import rtde_control

rtde_r = rtde_receive.RTDEReceiveInterface("10.33.55.90")
print("TCP Pose:", rtde_r.getActualTCPPose())

rtde_c = rtde_control.RTDEControlInterface("10.33.55.90")
print("RTDE Control connected!")
```
**Note**: `getActualTCPPose()` returns the pose of the active robot TCP. In this deployment workflow, TCP is treated as **always flange**.
If RTDE control fails, ensure the robot is in **Remote Control** and the motors are **ON** (brakes released).
Policy origin is set relative to flange via CLI (`--flange-to-policy-origin-m`).

---

## Step 8: Create Deployment Configuration

Create a configuration file or modify `ip/deployment/cli.py`:

```python
import numpy as np
from ip.deployment.config import (
    DeploymentConfig,
    CameraConfig,
    SegmentationConfig,
    GripperConfig,
    RTDEControlConfig,
)

# Camera transforms (from Step 5)
T_world_camera_cam1 = np.array([
    [0.9952, 0.0067, -0.0978, -0.4402],
    [-0.0729, -0.6154, -0.7849, 1.1759],
    [-0.0654, 0.7882, -0.6119, 0.0865],
    [0.0, 0.0, 0.0, 1.0],
])

T_world_camera_cam2 = np.array([
    [0.9978, 0.0556, -0.0352, -0.5160],
    [-0.0283, -0.1209, -0.9923, 1.5065],
    [-0.0595, 0.9911, -0.1190, -0.0123],
    [0.0, 0.0, 0.0, 1.0],
])

config = DeploymentConfig(
    # Cameras
    camera_configs=[
        CameraConfig(
            serial="f1380660",
            T_world_camera=T_world_camera_cam1,
            width=640,
            height=480,
            fps=30,
            align_to_color=True,
        ),
        CameraConfig(
            serial="f1371463",
            T_world_camera=T_world_camera_cam2,
            width=640,
            height=480,
            fps=30,
            align_to_color=True,
        ),
    ],
   
    # Robot
    robot_ip="10.33.55.90",
   
    # Model
    model_path="./checkpoints/ip",
    num_demos=2,
    num_traj_wp=10,
    num_diffusion_iters=4,
   
    # Segmentation
    segmentation=SegmentationConfig(
        enable=True,
        backend="xmem",
        sam_checkpoint_path="./checkpoints/sam/sam_vit_b_01ec64.pth",
        xmem_checkpoint_path="./checkpoints/xmem/XMem.pth",
        xmem_init_with_sam=True,
    ),
   
    # Gripper
    gripper=GripperConfig(
        enable=True,
        host=None,  # Uses robot_ip
        port=63352,
        open_position=0,
        closed_position=255,
    ),
   
    # Control
    rtde=RTDEControlConfig(
        control_mode="servoL",  # or "moveL"
        move_speed=0.25,          # moveL default [m/s]
        move_acceleration=1.2,    # moveL default [m/s^2]
        servo_time=0.002,         # e-Series servo period [s]
        servo_lookahead=0.1,      # [0.03, 0.2]
        servo_gain=300,           # [100, 2000]
    ),
   
    # Safety (per-step limits)
    safety=None,  # Uses defaults, or provide SafetyLimits
   
    # Execution
    execute_until_grip_change=True,
    # Runtime frame convention (default in this repo)
    tcp_offset_in_code=True,  # apply flange->policy-origin offset in code
    tcp_offset_m=np.array([0.0, 0.0, 0.088], dtype=np.float64),  # flange->policy-origin offset
    device="cuda:0",
)
```

### Waypoint selection (num_traj_wp)
Each recorded demo is **downsampled to `num_traj_wp` waypoints** before being fed to the model. The selection is **not uniform** in time. The current logic in `extract_waypoints()`:
- Always includes the **first** and **last** frame.
- First does a **motion compression pass** that removes near-static pause frames while preserving any gripper flip.
- Adds mandatory anchors at **gripper state transitions** and the frame just before each transition.
- Fills remaining slots to `num_traj_wp` by **farthest-point sampling on cumulative SE(3) arc length**, not by wall-clock time.

This avoids wasting waypoints on long pauses while still capturing task-stage boundaries and motion geometry.
**Important:** gripper-change waypoints are detected from the final binary `grips` signal.
`grips` is derived from measured gripper openness (`get_gripper_state`) using RLBench-style binarization:
open if `open_amount > 0.9`, else closed.

---

## Step 9: Collect Demonstrations

### 9.0 Home Position (default)
By default, `ip.deployment` will move the robot to a saved **home joint position** before starting demo collection or deployment.

If you haven't saved one yet:
```bash
python -m ip.deployment.utils.set_home_position --robot-ip <ROBOT_IP> --save-current
```
This writes `ip/deployment/assets/home_joint.json`.

To skip the home move, add `--no-home`.
By default, the gripper is also opened before starting. To skip this, add `--no-open-gripper`.

### 9.1 Start Demo Collection
```bash
python -m ip.deployment --collect-demo --demo-out demos/task1_demo1.pkl
```
For right-arm deployment, add `--arm right` so the default calibration file switches to `realsense_T_world_camera_right.json`.

Optional: save RGB+mask images for the selected waypoint frames (useful to verify `extract_waypoints()`):
```bash
python -m ip.deployment --collect-demo --demo-out demos/task1_demo1.pkl \
  --debug-demo-waypoints
```
This writes waypoint debug images to `ip/deployment/debug_waypoints` using `num_traj_wp` waypoints.

Frame metadata in saved demo:
- `frame_spec`: robot TCP frame (fixed flange) and `flange_to_policy_origin_m`.
- `recorded_at_utc`: demo timestamp.

Quickly inspect which frames were selected (prints indices + EE positions):
```bash
python -m ip.deployment.utils.inspect_demo --demo demos/task1_demo1.pkl
```

### 9.2 Recording Process
1. Move robot to start position
2. Press **ENTER** to begin (robot enters freedrive mode)
3. Kinesthetically demonstrate the task
4. Use `o` to open and `c` to close the gripper while recording
5. Press `q` or **ESC** to stop recording

### 9.3 Collect Multiple Demos
Repeat for 2-5 demonstrations per task:
```bash
python -m ip.deployment --collect-demo --demo-out demos/task1_demo2.pkl
python -m ip.deployment --collect-demo --demo-out demos/task1_demo3.pkl
```

---

## Step 10: Run Deployment

### 10.1 Single Demo
```bash
python -m ip.deployment --demo demos/task1_demo1.pkl
```

Defaults:
- executes predicted actions until gripper-state change (prevents gripper oscillation)
- `--flange-to-policy-origin-m 0 0 0.088` (default)
- `--arm left` (loads left-arm calibration by default)

Frame consistency check:
- Demos must include `frame_spec` with `robot_tcp_frame=flange` and `flange_to_policy_origin_m`.
- Deployment validates this strictly and fails on any mismatch or missing key.

Override example:
```bash
python -m ip.deployment --demo demos/task1_demo1.pkl --flange-to-policy-origin-m 0 0 0.088
```

### 10.2 Multiple Demos
```bash
python -m ip.deployment --demo demos/task1_demo1.pkl demos/task1_demo2.pkl
```

Convenient glob (bash will expand `*`):
```bash
python -m ip.deployment --demo demos/task1_demo*.pkl
```

### 10.3 Python API
```python
from ip.deployment.orchestrator import InstantPolicyDeployment
import pickle

# Load demos
demos = []
for path in ["demos/task1_demo1.pkl", "demos/task1_demo2.pkl"]:
    with open(path, "rb") as f:
        demos.append(pickle.load(f))

# Run deployment
deployment = InstantPolicyDeployment(config)
success = deployment.run(demos, max_steps=100)
print("Deployment success:", success)
```

### 10.4 Optional Live Outputs (Recommended for Audit)
Save the live rollout in demo-compatible format (`pcds`, `T_w_es`, `grips`):
```bash
python -m ip.deployment --demo demos/task1_demo1.pkl \
  --save-live
```

Replay the saved live rollout in EE-frame viewer:
```bash
python -m ip.deployment.utils.view_demo_pcds --demo ip/deployment/live.pkl
```

Save per-step live RGB+mask snapshots:
```bash
python -m ip.deployment --demo demos/task1_demo1.pkl \
  --debug-live-frames
```

Live debug image overlay fields:
- `step`: deployment step index
- `raw`: measured gripper openness
- `grip`: binarized gripper state (0/1)

Design rule:
- Debug images are saved as sidecar files only.
- `demo.pkl` / `live.pkl` stay focused on policy data (`pcds`, `T_w_es`, `grips`).

Deprecated/removed runtime debug paths:
- `--viz`
- `--viz-hz`
- `--record-live-pcd`
- `python -m ip.deployment.debug_segmentation`
- `python -m ip.deployment.debug_xmem_tracking`
- `python -m ip.deployment.view_demo_live_alignment`

---

## Debug Pipeline (recommended for every new setup)

### D1. Collect demo with optional waypoint RGB+mask export
```bash
python -m ip.deployment --collect-demo --demo-out demos/task1_demo1.pkl \
  --debug-demo-waypoints
```
This saves the selected waypoint frames as RGB+mask overlays for quick audit.

### D2. Deploy with optional live outputs
```bash
python -m ip.deployment --demo demos/task1_demo1.pkl \
  --save-live \
  --debug-live-frames
```
This produces:
- `ip/deployment/live.pkl` (demo-compatible rollout with `pcds`, `T_w_es`, `grips`)
- per-step RGB+mask overlays in `ip/deployment/debug_live_frames`
- `live.pkl` also includes `frame_spec` and `recorded_at_utc` metadata.

### D3. Replay `.pkl` point clouds in EE-frame viewer
```bash
python -m ip.deployment.utils.view_demo_pcds --demo demos/task1_demo1.pkl
python -m ip.deployment.utils.view_demo_pcds --demo ip/deployment/live.pkl
```
Open the URL printed in the terminal (Viser may auto-select `8080`, `8081`, ...).
Playback speed can be adjusted from the **FPS** slider in the viewer GUI.

---

## Quick Reference: File Locations

| Item              | Path                                                |
| ----------------- | --------------------------------------------------- |
| Deployment config | `ip/deployment/cli.py` or custom script             |
| Deployment docs   | `ip/deployment/docs/*.md`                           |
| Deployment utils  | `ip/deployment/utils/*.py`                          |
| Deployment assets | `ip/deployment/assets/*.json`                       |
| Calibration tools | `ip/deployment/calibration/*.py`                    |
| SAM checkpoint    | `checkpoints/sam/sam_vit_b_01ec64.pth`              |
| XMem++ checkpoint | `checkpoints/xmem/XMem.pth`                         |
| IP model          | `checkpoints/ip/{config.pkl, model.pt}`             |
| XMem++ source     | `XMem2-main/` (sibling to `ip/`)                    |
| Demos             | `demos/*.pkl`                                       |
