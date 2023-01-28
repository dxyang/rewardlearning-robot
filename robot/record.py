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
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tap import Tap

from robot import FrankaEnv
from robot.utils import HZ
from robot.joystick_controller import Buttons


class ArgumentParser(Tap):
    # fmt: off
    task: str                                           # Task ID for demonstration collection
    data_dir: Path = Path("data/demos/")                # Path to parent directory for saving demonstrations

    # Task Parameters
    include_visual_states: bool = False                 # Whether to run playback/get visual states (only False for now)
    max_time_per_demo: int = 15                         # Max time (in seconds) to record demo -- default = 21 seconds

    # Collection Parameters
    collection_strategy: str = "kinesthetic"            # How demos are collected :: only `kinesthetic` for now!
    controller: str = "joint"                           # Demonstration & playback uses a Joint Impedance controller...
    resume: bool = True                                 # Resume demonstration collection (on by default)
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

    # Initialize environment in `record` mode...
    print("[*] Initializing Robot Connection...")
    env = FrankaEnv(
        home="default",
        hz=HZ,
        controller=args.controller,
        mode="record",
        use_camera=False,
        use_gripper=False,
    )

    # Initializing Button Control... TODO(siddk) -- switch with ASR
    print("[*] Connecting to Button Controller...")
    buttons, demo_index = Buttons(), 1

    # If `resume` -- get "latest" index
    if args.resume:
        files = os.listdir(demo_rgb_dir) if args.include_visual_states else os.listdir(demo_raw_dir)
        if len(files) > 0:
            demo_index = max([int(x.split("-")[-1].split(".")[0]) for x in files]) + 1

    # Start Recording Loop
    print("[*] Starting Demo Recording Loop...")
    while True:
        print(f"[*] Starting to Record Demonstration `{demo_index}`...")
        demo_file = f"{args.task}-{datetime.now().strftime('%m-%d')}-{demo_index}.npz"

        # Set `record`
        env.set_mode("record")

        # Reset environment & wait on user input...
        env.reset()
        print(
            "[*] Ready to record!\n"
            f"\tYou have `{args.max_time_per_demo}` secs to complete the demo, and can use (X) to stop recording.\n"
            "\tPress (Y) to reset, and (A) to start recording!\n "
        )

        # Loop on valid button input...
        a, _, y = buttons.input()
        while not a and not y:
            a, _, y = buttons.input()

        # Reset if (Y)...
        if y:
            continue

        # Go, go, go!
        print("\t=>> Started recording... press (X) to terminate recording!")

        # Drop into Recording Loop --> for `record` mode, we really only care about joint positions
        #   =>> TODO(siddk) - handle Gripper?
        joint_qs = []
        for _ in range(int(args.max_time_per_demo * HZ) - 1):
            # Get Button Input (only if True) --> handle extended button press...
            _, x, _ = buttons.input()

            # Terminate...
            if x:
                print("\tHit (X) - stopping recording...")
                break

            # Otherwise no termination, keep on recording...
            else:
                obs, _, _, _ = env.step(None)
                joint_qs.append(obs["q"])

        # Close Environment
        env.close()

        # Save "raw" demonstration...
        np.savez(str(demo_raw_dir / demo_file), hz=HZ, qs=joint_qs)

        # Enter Phase 2 -- Playback (Optional if not `args.include_visual_states`)
        do_playback = True
        if args.include_visual_states:
            print("[*] Entering Playback Mode - Please reset the environment to beginning and get out of the way!")
        else:
            # Loop on valid user button input...
            print("[*] Optional -- would you like to replay demonstration? Press (A) to playback, and (X) to continue!")
            a, x, _ = buttons.input()
            while not a and not x:
                a, x, _ = buttons.input()

            # Skip Playback!
            if x:
                do_playback = False

        # Special Playback Handling -- change gains, and replay!
        if do_playback:
            # TODO(siddk) -- handle Camera observation logging...
            env.set_mode("default")
            env.reset()

            # Block on User Ready -- Robot will move, so this is for safety...
            print("\tReady to playback! Get out of the way, and hit (A) to continue...")
            a, _, _ = buttons.input()
            while not a:
                a, _, _ = buttons.input()

            # Execute Trajectory
            print("\tReplaying...")
            for idx in range(len(joint_qs)):
                env.step(joint_qs[idx])

            # Close Environment
            env.close()

        # Move on?
        print("Next? Press (A) to continue or (Y) to quit... or (X) to retry demo and skip save")

        # Loop on valid button input...
        a, x, y = buttons.input()
        while not a and not y and not x:
            a, x, y = buttons.input()

        # Exit...
        if y:
            break

        # Bump Index
        if not x:
            demo_index += 1

    # And... that's all folks!
    print("[*] Done Demonstrating -- Cleaning Up! Thanks 🤖🚀")


if __name__ == "__main__":
    demonstrate()