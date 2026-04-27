#!/usr/bin/env python3
"""Interactive 6-DoF IK control for the ARX X5 robot in PyBullet.

Keyboard controls (press/hold):
- Translation:  W/S (+/-X), A/D (+/-Y), R/F (+/-Z)
- Rotation:     U/O (+/-roll), I/K (+/-pitch), J/L (+/-yaw)
- Utility:      Z reset target pose, P print target+EE pose, Q quit
"""

from __future__ import annotations

import argparse
import math
import os
import time
from dataclasses import dataclass

import numpy as np
import pybullet as p
import pybullet_data


@dataclass
class RobotInfo:
    robot_id: int
    arm_joint_indices: list[int]
    arm_joint_names: list[str]
    lower_limits: list[float]
    upper_limits: list[float]
    joint_ranges: list[float]
    rest_poses: list[float]
    ee_link_index: int


def quat_mul(q1: tuple[float, float, float, float], q2: tuple[float, float, float, float]):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return (
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    )


def load_robot(urdf_path: str, use_fixed_base: bool = True) -> RobotInfo:
    flags = p.URDF_USE_SELF_COLLISION
    robot_id = p.loadURDF(urdf_path, [0, 0, 0], useFixedBase=use_fixed_base, flags=flags)

    n_joints = p.getNumJoints(robot_id)
    arm_joint_indices = []
    arm_joint_names = []
    lower_limits = []
    upper_limits = []
    rest_poses = []

    for i in range(n_joints):
        ji = p.getJointInfo(robot_id, i)
        joint_name = ji[1].decode("utf-8")
        joint_type = ji[2]

        # Arm DoFs: first six revolute joints in this URDF.
        if joint_type == p.JOINT_REVOLUTE and len(arm_joint_indices) < 6:
            arm_joint_indices.append(i)
            arm_joint_names.append(joint_name)
            lo = float(ji[8])
            hi = float(ji[9])
            if lo > hi:
                lo, hi = -math.pi, math.pi
            lower_limits.append(lo)
            upper_limits.append(hi)
            rest_poses.append(0.0)

        # Disable default motors on all controllable joints.
        if joint_type in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
            p.setJointMotorControl2(robot_id, i, p.VELOCITY_CONTROL, force=0)

    if len(arm_joint_indices) != 6:
        raise RuntimeError(
            f"Expected 6 revolute arm joints, found {len(arm_joint_indices)}."
        )

    ee_link_index = arm_joint_indices[-1]
    joint_ranges = [max(hi - lo, 1e-3) for lo, hi in zip(lower_limits, upper_limits)]

    return RobotInfo(
        robot_id=robot_id,
        arm_joint_indices=arm_joint_indices,
        arm_joint_names=arm_joint_names,
        lower_limits=lower_limits,
        upper_limits=upper_limits,
        joint_ranges=joint_ranges,
        rest_poses=rest_poses,
        ee_link_index=ee_link_index,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="ARX X5 interactive IK in PyBullet")
    parser.add_argument("--urdf", default="ARX-X5/X5A.urdf", help="Path to ARX X5 URDF")
    parser.add_argument("--dt", type=float, default=1.0 / 240.0, help="Simulation timestep")
    parser.add_argument(
        "--pos-step",
        type=float,
        default=0.004,
        help="Position increment (meters) per control tick",
    )
    parser.add_argument(
        "--rot-step-deg",
        type=float,
        default=1.2,
        help="Rotation increment (degrees) per control tick",
    )
    parser.add_argument(
        "--record-mp4",
        default="",
        help="Optional output MP4 path recorded by PyBullet (e.g. demo.mp4)",
    )
    return parser.parse_args()


def is_key_down(keys: dict[int, int], ch: str) -> bool:
    return any(
        ord(c) in keys and keys[ord(c)] & p.KEY_IS_DOWN for c in (ch.lower(), ch.upper())
    )


def is_key_triggered(keys: dict[int, int], ch: str) -> bool:
    return any(
        ord(c) in keys and keys[ord(c)] & p.KEY_WAS_TRIGGERED for c in (ch.lower(), ch.upper())
    )


def main():
    args = parse_args()
    urdf_path = os.path.abspath(args.urdf)
    if not os.path.exists(urdf_path):
        raise FileNotFoundError(f"URDF not found: {urdf_path}")

    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(args.dt)
    p.loadURDF("plane.urdf")

    info = load_robot(urdf_path)
    logger_id = None
    if args.record_mp4:
        output_path = os.path.abspath(args.record_mp4)
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        logger_id = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, output_path)
        print(f"Recording MP4 to: {output_path}")

    # Better camera view for desktop demo recording.
    p.resetDebugVisualizerCamera(
        cameraDistance=0.8,
        cameraYaw=45,
        cameraPitch=-30,
        cameraTargetPosition=[0.15, 0.0, 0.25],
    )

    ee_state = p.getLinkState(info.robot_id, info.ee_link_index, computeForwardKinematics=True)
    target_pos = np.array(ee_state[4], dtype=np.float64)
    target_quat = tuple(ee_state[5])

    rot_step_rad = math.radians(args.rot_step_deg)

    print("\n=== ARX X5 IK keyboard controls ===")
    print("Move : W/S (+/-X), A/D (+/-Y), R/F (+/-Z)")
    print("Rot  : U/O (+/-Roll), I/K (+/-Pitch), J/L (+/-Yaw)")
    print("Other: Z reset target pose, P print status, Q quit\n")
    print(f"Arm joints used for IK: {list(zip(info.arm_joint_indices, info.arm_joint_names))}")
    print(f"End-effector link index: {info.ee_link_index}\n")

    try:
        while True:
            keys = p.getKeyboardEvents()

            dpos = np.zeros(3, dtype=np.float64)
            droll = dpitch = dyaw = 0.0

            # Translation controls.
            if is_key_down(keys, "w"):
                dpos[0] += args.pos_step
            if is_key_down(keys, "s"):
                dpos[0] -= args.pos_step
            if is_key_down(keys, "a"):
                dpos[1] += args.pos_step
            if is_key_down(keys, "d"):
                dpos[1] -= args.pos_step
            if is_key_down(keys, "r"):
                dpos[2] += args.pos_step
            if is_key_down(keys, "f"):
                dpos[2] -= args.pos_step

            # Orientation controls.
            if is_key_down(keys, "u"):
                droll += rot_step_rad
            if is_key_down(keys, "o"):
                droll -= rot_step_rad
            if is_key_down(keys, "i"):
                dpitch += rot_step_rad
            if is_key_down(keys, "k"):
                dpitch -= rot_step_rad
            if is_key_down(keys, "j"):
                dyaw += rot_step_rad
            if is_key_down(keys, "l"):
                dyaw -= rot_step_rad

            if np.linalg.norm(dpos) > 0:
                target_pos += dpos

            if any(abs(v) > 0 for v in (droll, dpitch, dyaw)):
                delta_quat = p.getQuaternionFromEuler([droll, dpitch, dyaw])
                target_quat = quat_mul(delta_quat, target_quat)

            # Utility controls (on key trigger).
            if is_key_triggered(keys, "z"):
                ee_state = p.getLinkState(
                    info.robot_id, info.ee_link_index, computeForwardKinematics=True
                )
                target_pos = np.array(ee_state[4], dtype=np.float64)
                target_quat = tuple(ee_state[5])
                print("Reset target pose to current EE pose.")

            if is_key_triggered(keys, "p"):
                ee_state = p.getLinkState(
                    info.robot_id, info.ee_link_index, computeForwardKinematics=True
                )
                ee_pos = np.array(ee_state[4])
                ee_rpy = p.getEulerFromQuaternion(ee_state[5])
                tgt_rpy = p.getEulerFromQuaternion(target_quat)
                print("--- Status ---")
                print("Target pos:", np.round(target_pos, 4), "Target rpy:", np.round(tgt_rpy, 4))
                print("EE     pos:", np.round(ee_pos, 4), "EE     rpy:", np.round(ee_rpy, 4))

            if is_key_triggered(keys, "q"):
                break

            ik_solution = p.calculateInverseKinematics(
                info.robot_id,
                info.ee_link_index,
                targetPosition=target_pos.tolist(),
                targetOrientation=target_quat,
                lowerLimits=info.lower_limits,
                upperLimits=info.upper_limits,
                jointRanges=info.joint_ranges,
                restPoses=info.rest_poses,
                maxNumIterations=100,
                residualThreshold=1e-5,
            )

            for joint_idx in info.arm_joint_indices:
                target_angle = ik_solution[joint_idx]
                p.setJointMotorControl2(
                    bodyUniqueId=info.robot_id,
                    jointIndex=joint_idx,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=target_angle,
                    force=200,
                    positionGain=0.12,
                    velocityGain=1.0,
                )

            p.stepSimulation()
            time.sleep(args.dt)
    finally:
        if logger_id is not None:
            p.stopStateLogging(logger_id)
        p.disconnect()


if __name__ == "__main__":
    main()
