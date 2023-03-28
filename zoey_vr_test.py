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

from robot.data import RoboDemoDset
from robot.utils import HZ
from robot.vr import OculusController
from robot.xarm_env import XArmCentimeterSafeEnvironment


class ArgumentParser(Tap):
    # fmt: off
    task: str  # Task ID for demonstration collection
    data_dir: Path = Path("data/demos/")  # Path to parent directory for saving demonstrations
    include_visual_states: bool = True  # Whether to run playback/get visual states (only False for now)
    max_time_per_demo: int = 1000  # Max time (in seconds) to record demo -- default = 21 seconds
    random_reset: bool = False  # randomly initialize home pose with an offset
    resume: bool = True  # Resume demonstration collection (on by default)
    plot_trajectory: bool = False  # generate html plot for visualizing trajectory
    show_viewer: bool = False

def demonstrate() -> None:
    args = ArgumentParser().parse_args()
    print("[*] Initializing Robot Connection...")
    env = XArmCentimeterSafeEnvironment(
        control_frequency_hz=HZ,
        use_camera=False,
        # TODO: create some sort of button pressing mechanism to open and close the gripper,
        use_gripper=True,
        random_reset_home_pose=args.random_reset,
        speed=150,
        low_colision_sensitivity=True
    )
    oculus = OculusController()
    print("[*] Starting Demo Recording Loop...")
    while True:
        obs = env.reset()
        print(
            "[*] Ready to record!\n"
            f"\tYou have `{args.max_time_per_demo}` secs to complete the demo, and can use (Trigger) to stop recording.\n"
            "\tPress (B) to quit, and (A) to start recording!\n "
        )
        # Loop on valid button input...
        press = oculus.get_buttons()
        a = press['A']
        b = press['B']
        while not a and not b:
            press = oculus.get_buttons()
            a = press['A']
            b = press['B']

        if b:
            break
        oculus.reset()
        for _ in range(int(args.max_time_per_demo * HZ) - 1):
            press = oculus.get_buttons()
            stop = press['B']
            if stop:
                print("stopping recording...")
                break
            else:
                deltas = oculus.get_deltas()
                deltas[2] *= 2.5
                pick = press['RG']
                deltas = list(deltas)
                deltas.append(-1)
                deltas=np.array(deltas)
                if pick:
                    deltas[3] = 1
                deltas=2*deltas
                obs, _, _, _ = env.step(action=2*deltas, delta=True)
        env.close()

if __name__ == "__main__":
    demonstrate()
