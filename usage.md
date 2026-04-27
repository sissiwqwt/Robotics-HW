# ARX X5 PyBullet IK Interactive Control (Run Guide)

## 1) Install dependencies
```bash
python -m pip install pybullet numpy
```

## 2) Run the interactive controller
From the repository root:
```bash
python x5_ik_keyboard.py
```

Optional: record MP4 directly while you interact:
```bash
python x5_ik_keyboard.py --record-mp4 arx_x5_ik_demo.mp4
```

## 3) Keyboard mapping (6-DoF end-effector IK)
- **Translation**
  - `W / S`: `+x / -x`
  - `A / D`: `+y / -y`
  - `R / F`: `+z / -z`
- **Rotation**
  - `I / K`: `+roll / -roll`
  - `J / L`: `+pitch / -pitch`
  - `U / O`: `+yaw / -yaw`
- **Other**
  - `N / M`: gripper open / close
  - `T`: reset IK target to current end-effector pose
  - `Q`: quit

## 4) Export a short GIF from MP4 (optional)
If `ffmpeg` is installed:
```bash
ffmpeg -y -i arx_x5_ik_demo.mp4 -vf "fps=12,scale=960:-1:flags=lanczos" -t 6 arx_x5_ik_demo.gif
```

This creates a short ~6 second GIF suitable for assignment submission.

## 5) Notes
- The script parses `ARX-X5/X5A.urdf` first and prints a joint/link summary as the required URDF analysis step.
- Default end-effector frame is the child link of `joint6` (`link6`).
- You can change step sizes with:
```bash
python x5_ik_keyboard.py --pos-step 0.003 --rot-step 0.02
```
