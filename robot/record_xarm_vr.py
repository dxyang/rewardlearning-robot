"""
demonstrate.py

Collect interleaved demonstrations (in the case of kinesthetic teaching) of recording a kinesthetic demo,
then (optionally) playing back the demonstration to collect visual states.

As we're using Polymetis, you should use the following to launch the robot controller:
    > launch_robot.py --config-path /home/iliad/Projects/oncorr/conf/robot --config-name robot_launch.yaml timeout=15;
    > launch_gripper.py --config-path /home/iliad/Projects/oncorr/conf/robot --config-name gripper_launch.yaml;

References:
    - https://github.com/facebookresearch/fairo/blob/main/polymetis/examples/2_impedance_control.py
    - https://github.com/AGI-Labs/franka_control/blob/master/record.py
    - https://github.com/AGI-Labs/franka_control/blob/master/playback.py
"""
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from tap import Tap
from pynput.keyboard import Key, Listener
import sys
import select
import tty
import termios

from cam.utils import VideoRecorder

from robot.base_robot import XArmTaskSpaceEnv
from robot.data import RoboDemoDset
from robot.utils import HZ
from robot.vr import OculusController
from robot.xarm_env import XArmEnv


class ArgumentParser(Tap):
    # fmt: off
    task: str  # Task ID for demonstration collection
    data_dir: Path = Path("data/demos/")  # Path to parent directory for saving demonstrations


    # Task Parameters
    include_visual_states: bool = True  # Whether to run playback/get visual states (only False for now)
    max_time_per_demo: int = 21  # Max time (in seconds) to record demo -- default = 21 seconds

    random_reset: bool = False  # randomly initialize home pose with an offset

    # Collection Parameters
    resume: bool = True  # Resume demonstration collection (on by default)

    plot_trajectory: bool = False  # generate html plot for visualizing trajectory
    show_viewer: bool = False
    # fmt: on


def demonstrate() -> None:
    args = ArgumentParser().parse_args()

    # Make directories for "raw" recorded states, and playback RGB states...
    #   > Note: the "record" + "playback" split is use for "kinesthetic" demos for obtaining visual state w/o humans!
    demo_raw_dir = args.data_dir / args.task / "record-raw"
    os.makedirs(demo_raw_dir, exist_ok=args.resume)
    if args.include_visual_states:
        demo_rgb_dir = args.data_dir / args.task / "playback-rgb"
        os.makedirs(demo_rgb_dir, exist_ok=args.resume)

    # data saving yay
    video_recorder = VideoRecorder(save_dir=demo_rgb_dir, fps=HZ)
    h5py_dset_path = args.data_dir / args.task / "demos.hdf"
    dset = RoboDemoDset(save_path=h5py_dset_path, read_only_if_exists=False)

    # Initialize environment in `record` mode...
    print("[*] Initializing Robot Connection...")
    env = XArmTaskSpaceEnv(
        control_frequency_hz=HZ,
        use_camera=args.include_visual_states,
        # TODO: create some sort of button pressing mechanism to open and close the gripper,
        random_reset_home_pose=args.random_reset
    )
    oculus = OculusController()

    # If `resume` -- get "latest" index
    if args.resume:
        files = os.listdir(demo_rgb_dir) if args.include_visual_states else os.listdir(demo_raw_dir)
        if len(files) > 0:
            demo_index = max([int(x.split("-")[-1].split(".")[0]) for x in files]) + 1
        else:
            demo_index = 0

    # vc Start Recording Loop
    print("[*] Starting Demo Recording Loop...")
    while True:
        print(f"[*] Starting to Record Demonstration `{demo_index}`...")
        demo_file = f"{args.task}-{datetime.now().strftime('%m-%d')}-{demo_index}.npz"

        # Set `record`

        # Reset environment & wait on user input...
        obs = env.reset()
        print('here')
        # env.set_mode("record")

        print(
            "[*] Ready to record!\n"
            f"\tYou have `{args.max_time_per_demo}` secs to complete the demo, and can use (Trigger) to stop recording.\n"
            "\tPress (B) to quit, and (A) to start recording!\n "
        )
        # Loop on valid button input...

        press = oculus.get_buttons()
        a = press['A']
        b = press['B']
        tr = press['RTr']
        while not a and not tr and not b:
            press = oculus.get_buttons()
            a = press['A']
            b = press['B']
            tr = press['RTr']

        # Quit if (Y)...
        if b:
            break

        # Go, go, go!
        print("\t=>> Started recording... press (Trigger) to terminate recording!")

        # rgbs = []
        ee_poses = []
        ee_deltas = []
        oculus.reset()
        # angle_poses = []
        # print(args.max_time_per_demo)
        # print(HZ)
        for _ in range(int(args.max_time_per_demo * HZ) - 1):
            # visualize if camera
            if args.show_viewer:
                bgr = cv2.cvtColor(obs["rgb_image"], cv2.COLOR_RGB2BGR)
                cv2.imshow('RGB', bgr)
                cv2.waitKey(1)

            press = oculus.get_buttons()
            a = press['A']
            b = press['B']
            tr = press['RTr']

            # Get Button Input (only if True) --> handle extended button press...

            # Terminate...
            if tr:
                print("\tHit (Tr) - stopping recording...")
                break
            # Otherwise no termination, keep on recording...
            else:
                # dumb scaling because it works better
                deltas = oculus.get_deltas() *10
                deltas[2] *= 2.5
                #  print(deltas)
                # for i, delt in enumerate(deltas):
                #     if abs(delt) > 1:
                #         deltas[i] = 1 * delt / abs(delt)
                # print(deltas)
                # norm = np.linalg.norm(deltas)
                # if not norm == 0:
                #     deltas =  deltas / norm
                # print(deltas)
                obs, _, _, _ = env.step(action=deltas, delta=True)
                ee_poses.append(obs["ee_pos"])
                ee_deltas.append(obs['delta_ee_pos'])
                # angle_poses.append(obs['q'])
                # rgbs.append(obs["rgb_image"])


        env.close()

        # Save "raw" demonstration...
        np.savez(str(demo_raw_dir / demo_file), hz=HZ, ee_poses=ee_poses)

        # Enter Phase 2 -- Playback (Optional if not `args.include_visual_states`)
        do_playback = True
        if args.include_visual_states:
            print("[*] Entering Playback Mode - Please reset the environment to beginning and get out of the way!")
        else:
            # Loop on valid user button input...
            print("[*] Optional -- would you like to replay demonstration? Press (A) to playback, and (Tr) to continue!")
            press = oculus.get_buttons()
            a = press['A']
            b = press['B']
            tr = press['RTr']
            while not a and not x:
                press = oculus.get_buttons()
                a = press['A']
                b = press['B']
                tr = press['RTr']

            # Skip Playback!
            if tr:
                do_playback = False

        # Special Playback Handling -- change gains, and replay!
        jas = []
        eef_poses = []
        rgbs = []
        if do_playback:
            # TODO(siddk) -- handle Camera observation logging...
            obs = env.reset()
            # eef_poses.append(obs["ee_pose"].copy())
            # eef_xyzs.append(obs["ee_pose"][:3].copy())
            # jas.append(obs["q"].copy())
            rgbs.append(obs["rgb_image"].copy())
            # jas.append(obs['q'])
            eef_poses.append(obs['ee_pos'])

            # Block on User Ready -- Robot will move, so this is for safety...
            print("\tReady to playback! Get out of the way, and hit (A) to continue or TR to skip...")
            press = oculus.get_buttons()
            a = press['A']
            b = press['B']
            tr = press['RTr']
            while not a and not tr:
                press = oculus.get_buttons()
                a = press['A']
                b = press['B']
                tr = press['RTr']

            # Execute Trajectory
            print("\tReplaying...")
            for idx in range(len(ee_deltas)):
                if(tr):
                    break
                obs, _, _, _ = env.step(ee_deltas[idx], delta=True)
                rgbs.append(obs["rgb_image"].copy())
                # jas.append(obs['q'])
                eef_poses.append(obs['ee_pos'])
            # Close Environment
            env.close()

        if args.plot_trajectory:
            from viz.plot import PlotlyScene, plot_transform, plot_points_sequence
            scene = PlotlyScene(
                size=(600, 600), x_range=(-1, 1), y_range=(-1, 1), z_range=(-1, 1)
            )
            plot_transform(scene.figure, np.eye(4), label="world origin")
            # eef_xyzs_np = np.array(eef_xyzs).T # N x 3
            plot_points_sequence(scene.figure, points=eef_xyzs_np)
            scene.plot_scene_to_html("test")

        print(len(rgbs))

        # Move on?
        print("Next? Press (A) to save and continue or (B) to quit without saving or (Tr) to retry demo and skip save")

        # Loop on valid button input...        
        press = oculus.get_buttons()
        a = press['A']
        b = press['B']
        tr = press['RTr']
        while not a and not b and not tr:
            press = oculus.get_buttons()
            a = press['A']
            b = press['B']
            tr = press['RTr']
        # Exit...
        if b:
            break

        # Bump Index
        if not tr:
            for frame_idx, rgb_frame in enumerate(rgbs):
                if frame_idx == 0:
                    video_recorder.init(rgb_frame)
                else:
                    video_recorder.record(rgb_frame)
            save_str = str(demo_index).zfill(3)
            video_recorder.save(f"{save_str}.mp4")

            rgb_np = np.expand_dims(np.array(rgbs), axis=0)  # 1, horizon, h, w, c
            ja_np = np.expand_dims(np.array(jas), axis=0)  # 1, horizon, 7
            eefpose_np = np.expand_dims(np.array(eef_poses), axis=0)  # 1, horizon, 7
            print(rgb_np.shape)
            dset.add_traj(rgbs=rgb_np, joint_angles=ja_np, eef_poses=eefpose_np)  # TODO add  joint_angles=ja_np bcak in
            demo_index += 1

    # And... that's all folks!
    print("[*] Done Demonstrating -- Cleaning Up! Thanks 🤖🚀")


if __name__ == "__main__":
    demonstrate()
