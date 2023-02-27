"""
record_points.py

Use robot as a measuring stick. Useful for measuring constraints for safe robot maneuvering.
"""
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from tap import Tap

from cam.utils import VideoRecorder

from robot.env import FrankaEnv
from robot.data import RoboDemoDset
from robot.utils import HZ
from robot.joystick_controller import Buttons


class ArgumentParser(Tap):
    # fmt: off
    task: str = "workspace_measure"                     # Task ID for demonstration collection
    data_dir: Path = Path("data/demos/")                # Path to parent directory for saving demonstrations

    use_gripper: bool = False
    stream_points: bool = False

    # Task Parameters
    max_time_per_demo: int = 60                         # Max time (in seconds) to record demo -- default = 21 seconds

    # Collection Parameters
    collection_strategy: str = "kinesthetic"            # How demos are collected :: only `kinesthetic` for now!
    controller: str = "joint"                           # Demonstration & playback uses a Joint Impedance controller...
    resume: bool = True                                 # Resume demonstration collection (on by default)

    plot_trajectory: bool = True                       # generate html plot for visualizing trajectory
    show_viewer: bool = True
    # fmt: on


def demonstrate() -> None:
    args = ArgumentParser().parse_args()

    # Make directories for "raw" recorded states, and playback RGB states...
    #   > Note: the "record" + "playback" split is use for "kinesthetic" demos for obtaining visual state w/o humans!
    demo_raw_dir = args.data_dir / args.task / "record-raw"
    os.makedirs(demo_raw_dir, exist_ok=args.resume)

    # Initialize environment in `record` mode...
    print("[*] Initializing Robot Connection...")
    env = FrankaEnv(
        home="default",
        hz=HZ,
        controller=args.controller,
        mode="record",
        use_camera=True,
        use_gripper=args.use_gripper, # TODO: create some sort of button pressing mechanism to open and close the gripper,
        random_reset_home_pose=False
    )

    print("[*] Connecting to Button Controller...")
    buttons, demo_index = Buttons(), 1

    # If `resume` -- get "latest" index
    if args.resume:
        files = os.listdir(demo_raw_dir)
        if len(files) > 0:
            demo_index = max([int(x.split("-")[-1].split(".")[0]) for x in files]) + 1

    # Start Recording Loop
    print("[*] Starting Demo Recording Loop...")
    while True:
        print(f"[*] Starting to Record Demonstration `{demo_index}`...")
        demo_str = f"{args.task}-{datetime.now().strftime('%m-%d')}-{demo_index}"
        demo_file = f"{demo_str}.npz"

        # Set `record`
        env.set_mode("record")

        # Reset environment & wait on user input...
        obs = env.reset()
        print(
            "[*] Ready to record!\n"
            f"\tYou have `{args.max_time_per_demo}` secs to complete the demo, and can use (X) to stop recording.\n"
            "\tPress (Y) to quit, and (A) to start recording!\n "
        )

        # Loop on valid button input...
        a, _, x, y = buttons.input()
        while not a and not y:
            a, _, _, y = buttons.input()

        # Quit if (Y)...
        if y:
            break

        # Go, go, go!
        print("\t=>> Started recording... press (X) to terminate recording!")

        # Drop into Recording Loop --> for `record` mode, we really only care about joint positions
        #   =>> TODO(siddk) - handle Gripper?
        joint_qs = []
        gripper_controls = []
        eef_poses, eef_xyzs = [], []
        for _ in range(int(args.max_time_per_demo * HZ) - 1):
            # visualize if camera
            if args.show_viewer:
                bgr = cv2.cvtColor(obs["rgb_image"], cv2.COLOR_RGB2BGR)
                cv2.imshow('RGB', bgr)
                cv2.waitKey(1)

            # Get Button Input (only if True) --> handle extended button press...
            _, b, x, _ = buttons.input()

            # Terminate...
            if x:
                print("\tHit (X) - stopping recording...")
                break
            # Otherwise no termination, keep on recording...
            else:
                close_gripper = False
                if b:
                    close_gripper = True
                obs, _, _, _ = env.step(None, open_gripper = not close_gripper)
                joint_qs.append(obs["q"])
                gripper_controls.append(b)
                eef_poses.append(obs["ee_pose"].copy())
                eef_xyzs.append(obs["ee_pose"][:3].copy())
                if args.stream_points:
                    print(f"{eef_xyzs[-1]}")

        # Close Environment
        env.close()

        # Save "raw" demonstration...
        np.savez(str(demo_raw_dir / demo_file), hz=HZ, qs=joint_qs, eef_poses=eef_poses, eef_xyzs=eef_xyzs)

        if args.plot_trajectory:
            from viz.plot import PlotlyScene, plot_transform, plot_points_sequence
            scene = PlotlyScene(
                size=(600, 600), x_range=(-1, 1), y_range=(-1, 1), z_range=(-1, 1)
            )
            plot_transform(scene.figure, np.eye(4), label="world origin")
            eef_xyzs_np = np.array(eef_xyzs).T # N x 3
            plot_points_sequence(scene.figure, points=eef_xyzs_np)
            scene.plot_scene_to_html(str(demo_raw_dir / f"{demo_str}.html"))

        # Move on?
        print("Next? Press (A) to save and continue or (Y) to quit without saving or (X) to retry demo and skip save")

        # Loop on valid button input...
        a, _, x, y = buttons.input()
        while not a and not y and not x:
            a, _, x, y = buttons.input()

        # Exit...
        if y:
            break

    # And... that's all folks!
    print("[*] Done Demonstrating -- Cleaning Up! Thanks 🤖🚀")


if __name__ == "__main__":
    demonstrate()
