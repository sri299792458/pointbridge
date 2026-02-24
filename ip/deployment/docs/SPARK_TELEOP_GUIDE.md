# SPARK Demo Input (Unified Deployment Pipeline)

SPARK is now only a control input mode for demo collection.

There is one shared pipeline for:
- camera capture and segmentation
- robot state reads
- demo frame schema and waypoint debug export

The only difference is how motion/gripper commands are generated:
- `keyboard`: pendant freedrive + `o/c` hotkeys
- `spark`: Spark serial stream mapped to UR `servoJ` + gripper

In Spark mode, the alignment GUI stays open throughout demo collection and shows Spark-vs-UR pose alignment and bounds.

## Install
```bash
pip install -e ".[deployment]"
```

## Collect Demo with Spark Input
```bash
python -m ip.deployment \
  --collect-demo \
  --demo-out demos/task1_demo1.pkl \
  --demo-control spark \
  --spark-serial /dev/ttyUSB0 \
  --spark-profile lightning
```

Useful options (same shared collector path):
- `--camera-serial ...` filter cameras
- `--manual-seed` for manual XMem seeding
- `--debug-demo-waypoints` to save waypoint RGB+mask overlays

## Spark Flags
- `--spark-serial`: required when `--demo-control spark`
- `--spark-profile`: `lightning` or `thunder` (offset + gripper mapping defaults)
- `--spark-offsets-pickle`: optional override for offsets file path

## Notes
- Spark mode does not use pendant freedrive.
- GUI runs throughout collection:
  - `Start Recording` begins capture.
  - `Stop + Save` ends and saves.
  - `Cancel` or window close discards recording.
- Demo output format is the same as keyboard mode.
- If a configured offsets pickle path is missing, collection fails fast.
