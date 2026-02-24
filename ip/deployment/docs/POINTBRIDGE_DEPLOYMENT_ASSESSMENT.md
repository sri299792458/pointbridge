# PointBridge Deployment Assessment for IP Setup

## Scope

Assess whether `pointbridge/` can be deployed in the current `instant_policy/ip/deployment` stack, and define an execution plan for a production-grade integration.

## What We Have Now (IP Deployment)

- Robot/control stack is UR5e + Robotiq + RTDE control/state.
- Perception is RealSense depth + optional SAM/XMem segmentation -> fused world-frame point cloud.
- Demo capture stores `pcds`, `T_w_es`, `grips`, `grip_cmds`, `grip_raws` in pickle.
- Runtime policy loop already supports safe bounded execution and gripper commands.

Reference files:
- `instant_policy/ip/deployment/docs/DEPLOYMENT_GUIDE.md`
- `instant_policy/ip/deployment/perception/realsense_perception.py`
- `instant_policy/ip/deployment/demo/demo_collector.py`
- `instant_policy/ip/deployment/state/ur_rtde_state.py`

## What PointBridge Expects

- Primary real-world path in repo is FR3-focused (`franka_env`/Franka stack).
- VLM scene filtering pipeline assumes Molmo + SAM2 + FoundationStereo services.
- FR3 preprocessing scripts expect FrankaTeach-style raw data layout, then convert to PointBridge pkl format.
- Policy operates on robot points + object points + proprio in robot-base frame and predicts pose/gripper actions.

Reference files:
- `pointbridge/docs/real_evaluation.md`
- `pointbridge/point_bridge/suite/fr3.py`
- `pointbridge/point_bridge/robot_utils/fr3/process_data.py`
- `pointbridge/point_bridge/robot_utils/fr3/generate_pkl.py`
- `pointbridge/point_bridge/read_data/fr3.py`

## Critical Compatibility Result

PointBridge is **not drop-in deployable** on current IP deployment as-is. The gap is not just hardware naming; it spans data schema, perception assumptions, runtime wrappers, and some internal repo consistency issues.

## Concrete Blockers Found in PointBridge Repo

1. Packaging mismatch in `setup.py`.
- File filters `find_packages()` for `pointbridge*`, while actual packages are `point_bridge*`.
- File: `pointbridge/setup.py`

2. `config_eval.yaml` defaults point to missing config groups.
- Uses `agent: point_policy`, `suite: point_policy`, `dataloader: point_policy`.
- File: `pointbridge/point_bridge/cfgs/config_eval.yaml`

3. FR3 pixel key mismatch.
- `suite/fr3.yaml` uses `pixels_left/pixels_right`.
- FR3 camera mapping utility uses `pixels8_left/pixels8_right`.
- Files:
  - `pointbridge/point_bridge/cfgs/suite/fr3.yaml`
  - `pointbridge/point_bridge/robot_utils/fr3/utils.py`

4. Broken/fragile import and path assumptions in FR3 suite.
- Import missing `point_bridge.` prefix for gripper points.
- Calibration path references `robot_utils/fr3_nyu/...` while repo path is `robot_utils/fr3/...`.
- Server deploy import path references `model_servers.robot_fr3_nyu.client` (not present in tree).
- File: `pointbridge/point_bridge/suite/fr3.py`

5. Hardcoded placeholder/config requirements for VLM stack.
- `local.yaml` is placeholder path and must be set.
- Gemini client API key placeholder is hardcoded string.
- FoundationStereo server path is hardcoded to an absolute local path.
- Files:
  - `pointbridge/point_bridge/cfgs/local.yaml`
  - `pointbridge/point_bridge/detection_utils/utils.py`
  - `pointbridge/point_bridge/model_servers/foundation_stereo/server.py`

6. FR3 data scripts are template-like with hardcoded dataset roots and task names.
- File: `pointbridge/point_bridge/robot_utils/fr3/generate_pkl.py`

## Deployment Strategy for Our Setup

Use PointBridge model and training pipeline, but replace FR3-specific deployment/data interfaces with IP-native adapters.

### Phase 0: Stabilize PointBridge Fork (must do first)

- Fix packaging and config defaults so train/eval entrypoints are executable.
- Normalize FR3 pixel key naming and broken imports/paths.
- Remove hardcoded server/model paths and API keys; use env vars + config.
- Add smoke tests:
  - config composition
  - dataset loader construction
  - one forward pass with synthetic observation

### Phase 1: Data Interop Layer (IP demos -> PointBridge dataset)

Build converter script in IP repo, not ad-hoc notebooks.

Input:
- `instant_policy` demo pickle (`pcds`, `T_w_es`, `grips`, optional debug RGB/masks)

Output (PointBridge-compatible task `.pkl`):
- `observations`: list of trajectories with
  - `eef_states` as `[x,y,z,qx,qy,qz,qw]`
  - `gripper_states` as binary/open metric
  - `robot_tracks_3d` (UR keypoints)
  - `object_tracks_128_3d` (sampled from segmented object cloud)
  - optional image keys
- `task_emb`
- `camera_intrinsics`, `camera_extrinsics`

Notes:
- This bypasses FrankaTeach raw-data assumptions in `process_data.py`.
- Initial version can support single-object tracking; multitask/object-disambiguation comes later.

### Phase 2: UR Runtime Adapter for PointBridge

Create `suite/ur_ip.py` (or equivalent in IP deployment) that provides PointBridge-required observation dict from live IP sensors.

At each step:
- Build `robot_tracks_3d` from UR flange pose + UR gripper keypoint template.
- Build `object_tracks_128_3d` from segmented RealSense cloud in robot/world frame.
- Build `proprioceptive` and optional `task_emb`.
- Feed to PointBridge `BCAgent.act`.
- Map output action `(pos, rot6d, grip)` to UR RTDE command path and existing safety executor.

### Phase 3: Training and Evaluation on Our Data

- Train PointBridge on converted UR dataset.
- Use points + proprio first (`obs_type=[points]`, `action_mode=pose`).
- Validate offline replay before robot-in-the-loop.
- Then enable live deployment with conservative safety bounds and low horizon.

### Phase 4: Optional VLM Upgrade

- If needed, replace current SAM/XMem object selection with PointBridge VLM pipeline (Molmo/SAM2/FoundationStereo).
- Keep this optional because it adds operational complexity and separate service dependencies.

## Practical Risks

- Pretrained FR3 PointBridge checkpoints are unlikely to transfer directly to UR5e due to robot geometry/action distribution mismatch.
- FoundationStereo + Molmo services can become deployment bottlenecks unless tightly managed.
- Data quality (consistent object masks and calibrated frame transforms) is the largest determinant of success.

## Recommended Immediate Next Actions

1. Patch PointBridge repo blockers (Phase 0) in our local fork.
2. Implement a first-pass dataset converter from IP demo pkl to PointBridge pkl schema.
3. Train a small single-task PointBridge model on converted data and validate on recorded rollouts.
4. Integrate PointBridge policy backend into `ip/deployment/orchestrator.py` behind a feature flag.
