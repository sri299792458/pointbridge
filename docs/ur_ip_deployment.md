# PointBridge Deployment on UR + RealSense (No Training)

This is a deployment-only path for your UR + RealSense setup, with all runtime code under `pointbridge/`.

## Scope

- Uses PointBridge policy inference (`BCAgent` + checkpoint `stats`).
- Uses the vendored `ip/deployment` stack for UR RTDE control + RealSense capture.
- Does **not** use XMem.
- Does **not** require SAM/Molmo/FoundationStereo for this minimal path.

## Required Inputs

1. PointBridge policy checkpoint (`snapshot.pt`) that contains model weights + `stats`.
2. RealSense calibration JSON with `cameras.{serial}.T_world_camera`.
3. Robot network info (`--robot-ip`) and optional gripper connectivity.

## Checkpoint Availability (Important)

PointBridge repo/docs expect you to provide `bc_weight` / checkpoint, but do not include a built-in pretrained policy download script.
If you do not already have a compatible `snapshot.pt`, deployment cannot proceed yet.

## Run Deployment

```bash
cd pointbridge

python point_bridge/deploy_ur_ip.py \
  --checkpoint /path/to/snapshot.pt \
  --calibration /path/to/realsense_T_world_camera.json \
  --robot-ip 10.33.55.90 \
  --max-steps 200 \
  --x-min 0.15 --x-max 0.85 \
  --y-min -0.45 --y-max 0.45 \
  --z-min -0.05 --z-max 0.60
```

If your checkpoint is language-conditioned:

```bash
python point_bridge/deploy_ur_ip.py \
  --checkpoint /path/to/snapshot.pt \
  --calibration /path/to/realsense_T_world_camera.json \
  --task-text "put the bowl on the plate"
```

## Optional PointBridge-Native VLM Stack (Future)

For the paper-faithful real-world extraction path (Molmo + SAM2 + FoundationStereo), use PointBridge docs:

- `docs/install.md`
- `docs/real_evaluation.md`

Helper script:

```bash
bash scripts/download_deploy_models.sh
```

This helper only prepares PointBridge-native third-party model assets (SAM2 checkpoints and pointers for Molmo/FoundationStereo). It does not provide policy checkpoints.

## Files

- Deployment runner: `point_bridge/deploy_ur_ip.py`
- Vendored runtime stack: `ip/deployment/`
