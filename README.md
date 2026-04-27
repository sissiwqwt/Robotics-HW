# Robotics HW: Exponential Coordinates + ARX X5 IK Demo

This repository now contains:

1. **Foundational kinematics (`kin_func_skeleton.py`)** implemented with NumPy:
   - 3D skew operator `skew_3d`
   - axis-angle rotation matrix `rotation_3d`
   - twist hat map `hat_3d`
   - homogeneous transform from twist exponential `homog_3d`
   - product of exponentials `prod_exp`

2. **Interactive IK simulation (`x5_ik_interactive.py`)** for the ARX X5 URDF in PyBullet:
   - Loads `ARX-X5/X5A.urdf`
   - Uses inverse kinematics to command the 6-DoF arm (joints 1–6)
   - Real-time keyboard control of end-effector translation + orientation

---

## 1) Environment setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy pybullet
```

---

## 2) Run basic kinematics tests

```bash
python kin_func_skeleton.py
```

This runs the built-in numerical checks for the implemented 3D functions.

---

## 3) Run interactive ARX X5 IK simulation

```bash
python x5_ik_interactive.py
```

### Keyboard controls

- **Translation**
  - `W/S`: +X / -X
  - `A/D`: +Y / -Y
  - `R/F`: +Z / -Z
- **Rotation**
  - `U/O`: +roll / -roll
  - `I/K`: +pitch / -pitch
  - `J/L`: +yaw / -yaw
- **Other**
  - `Z`: reset target pose to current end-effector pose
  - `P`: print target and current EE pose
  - `Q`: quit

Optional arguments:

```bash
python x5_ik_interactive.py --pos-step 0.003 --rot-step-deg 1.0 --dt 0.0041667
```

---

## 4) Export a short demo video / GIF

### Option A: direct MP4 recording (recommended)

PyBullet can record video directly:

```bash
python x5_ik_interactive.py --record-mp4 demo_x5.mp4
```

Operate the arm with the keyboard for a few seconds, then press `Q` to end and finalize the video.

### Option B: convert MP4 to GIF

If you have `ffmpeg`:

```bash
ffmpeg -i demo_x5.mp4 -vf "fps=15,scale=960:-1:flags=lanczos" demo_x5.gif
```

---

## Notes

- The ARX URDF includes additional gripper prismatic joints (`joint7`, `joint8`).
  This demo solves IK for the **6 revolute arm joints** and uses link6 as the end-effector link.
- The script prints the exact arm joint indices/names at startup for transparency.
