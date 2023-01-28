"""
Gym interface to interact with the real robot
"""
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import grpc
import numpy as np
import torch
import gym
from polymetis import GripperInterface, RobotInterface
from scipy.spatial.transform import Rotation as R


from robot.controllers import ResolvedRateControl
from robot.util import (
    ROBOT_IP, HOMES, HZ,
    KQ_GAINS, KQD_GAINS, KX_GAINS, KXD_GAINS, KRR_GAINS,
    GRIPPER_MAX_WIDTH, GRIPPER_FORCE, GRIPPER_SPEED,
    Rate
)

# Silence OpenAI Gym warnings
gym.logger.setLevel(logging.ERROR)

"""
wrapper that addds the resolved rate controller + changes quaternion order to xyzw
"""
class RobotInterfaceWrapper(RobotInterface):
    def get_ee_pose(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Polymetis defaults to returning a Tuple of (position, orientation), where orientation is a quaternion in
        *scalar-first* format (w, x, y, z). However, `scipy` and other libraries expect *scalar-last* (x, y, z, w);
        we take care of that here!

        :return Tuple of (3D position, 4D orientation as a quaternion w/ scalar-last)
        """
        pos, quat = super().get_ee_pose()
        return pos, torch.roll(quat, -1)

    def start_resolved_rate_control(self, Kq: Optional[List[float]] = None) -> List[Any]:
        """
        Start Resolved-Rate Control (P control on Joint Velocity), as a non-blocking controller.

        The desired EE velocities can be updated using `update_desired_ee_velocities` (6-DoF)!
        """
        torch_policy = ResolvedRateControl(
            Kp=self.Kqd_default if Kq is None else Kq,
            robot_model=self.robot_model,
            ignore_gravity=self.use_grav_comp,
        )
        return self.send_torch_policy(torch_policy=torch_policy, blocking=False)

    def update_desired_ee_velocities(self, ee_velocities: torch.Tensor):
        """
        Update the desired end-effector velocities (6-DoF x, y, z, roll, pitch, yaw).

        Requires starting a resolved-rate controller via `start_resolved_rate_control` beforehand.
        """
        try:
            update_idx = self.update_current_policy({"ee_velocity_desired": ee_velocities})
        except grpc.RpcError as e:
            print(
                "Unable to update desired end-effector velocities. Use `start_resolved_rate_control` to start a "
                "resolved-rate controller."
            )
            raise e
        return update_idx


# === Polymetis Environment Wrapper ===
class FrankaEnv(gym.Env):
    def __init__(
        self,
        home: str = "default",
        hz: int = HZ,
        controller: str = "cartesian",
        mode: str = "default",
        use_camera: bool = False,
        use_gripper: bool = False,
    ) -> None:
        """
        Initialize a *physical* Franka Environment, with the given home pose, PD controller gains, and camera.

        :param home: Default home position (specified as a string index into `HOMES` above!
        :param hz: Default policy control hz; somewhere between 20-40 is a good range.
        :param controller: Which impedance controller to use in < joint | cartesian | osc >
        :param mode: Mode in < "default" | ... > --  used to set P(D) gains!
        :param use_camera: Boolean whether to initialize the Camera connection for recording visual states (WIP)
        :param use_gripper: Boolean whether to initialize the Gripper controller (default: False)
        """
        self.home = home
        self.rate = Rate(hz)
        self.mode = mode
        self.controller = controller
        self.curr_step = 0

        self.current_joint_pose, self.current_ee_pose, self.current_ee_rot = None, None, None
        self.robot, self.kp, self.kpd = None, None, None
        self.use_gripper, self.gripper, self.current_gripper_state, self.gripper_is_open = use_gripper, None, None, True

        if use_camera:
            raise NotImplementedError("Camera support not yet implemented!")

        # Initialize Robot and PD Controller
        self.reset()

    def robot_setup(self, home: str = "default", franka_ip: str = ROBOT_IP) -> None:
        # Initialize Robot Interface and Reset to Home
        self.robot = RobotInterfaceWrapper(ip_address=franka_ip)
        self.robot.set_home_pose(torch.Tensor(HOMES[home]))
        print(f"Robot going home!")
        self.robot.go_home()

        # Initialize current joint & EE poses...
        self.current_ee_pose = np.concatenate([a.numpy() for a in self.robot.get_ee_pose()])
        self.current_ee_rot = R.from_quat(self.current_ee_pose[3:]).as_euler("xyz")
        self.current_joint_pose = self.robot.get_joint_positions().numpy()

        # Create an *Impedance Controller*, with the desired gains...
        #   > Note: Feel free to add any other controller, e.g., a PD controller around joint poses.
        #           =>> Ref: https://github.com/AGI-Labs/franka_control/blob/master/util.py#L74
        if self.controller == "joint":
            # Note: P/D values of "None" default to... well the "default" values above 😅
            #   > These values are defined in the default launch_robot YAML (`robot_client/franka_hardware.yaml`)
            self.robot.start_joint_impedance(Kq=self.kp, Kqd=self.kpd)

        elif self.controller == "cartesian":
            # Note: P/D values of "None" default to... well the "default" values above 😅
            #   > These values are defined in the default launch_robot YAML (`robot_client/franka_hardware.yaml`)
            self.robot.start_cartesian_impedance(Kx=self.kp, Kxd=self.kpd)

        elif self.controller == "resolved-rate":
            # Note: P/D values of "None" default to... well the "default" values for Joint PD Control above 😅
            #   > These values are defined in the default launch_robot YAML (`robot_client/franka_hardware.yaml`)
            self.robot.start_resolved_rate_control(Kq=self.kp)

        else:
            raise NotImplementedError(f"Support for controller `{self.controller}` not yet implemented!")

        # Initialize Gripper Interface and Open
        if self.use_gripper:
            self.gripper = GripperInterface(ip_address=franka_ip)
            self.gripper.goto(GRIPPER_MAX_WIDTH, speed=GRIPPER_SPEED, force=GRIPPER_FORCE)
            gripper_state = self.gripper.get_state()
            self.current_gripper_state = {"width": gripper_state.width, "max_width": gripper_state.max_width}
            self.gripper_is_open = True


    def reset(self) -> Dict[str, np.ndarray]:
        # Set PD Gains -- kp, kpd -- depending on current mode, controller
        if self.controller == "joint":
            self.kp, self.kpd = KQ_GAINS[self.mode], KQD_GAINS[self.mode]
        elif self.controller == "cartesian":
            self.kp, self.kpd = KX_GAINS[self.mode], KXD_GAINS[self.mode]
        elif self.controller == "resolved-rate":
            self.kp = KRR_GAINS[self.mode]

        # Call setup with the new controller...
        self.robot_setup()
        return self.get_obs()

    def set_mode(self, mode: str) -> None:
        self.mode = mode

    def get_obs(self) -> Dict[str, np.ndarray]:
        new_joint_pose = self.robot.get_joint_positions().numpy()
        new_ee_pose = np.concatenate([a.numpy() for a in self.robot.get_ee_pose()])
        new_ee_rot = R.from_quat(new_ee_pose[3:]).as_euler("xyz")

        if self.use_gripper:
            new_gripper_state = self.gripper.get_state()

            # Note that deltas are "shifted" 1 time step to the right from the corresponding "state"
            obs = {
                "q": new_joint_pose,
                "qdot": self.robot.get_joint_velocities().numpy(),
                "delta_q": new_joint_pose - self.current_joint_pose,
                "ee_pose": new_ee_pose,
                "delta_ee_pose": new_ee_pose - self.current_ee_pose,
                "gripper_width": new_gripper_state.width,
                "gripper_max_width": new_gripper_state.max_width,
                "gripper_open": self.gripper_is_open,
            }

            # Bump "current" poses...
            self.current_joint_pose, self.current_ee_pose = new_joint_pose, new_ee_pose
            self.current_gripper_state = {"width": new_gripper_state.width, "max_width": new_gripper_state.max_width}
            return obs

        else:
            # Note that deltas are "shifted" 1 time step to the right from the corresponding "state"
            obs = {
                "q": new_joint_pose,
                "qdot": self.robot.get_joint_velocities().numpy(),
                "delta_q": new_joint_pose - self.current_joint_pose,
                "ee_pose": new_ee_pose,
                "delta_ee_pose": new_ee_pose - self.current_ee_pose,
            }

            # Bump "current" trackers...
            self.current_joint_pose, self.current_ee_pose, self.current_ee_rot = new_joint_pose, new_ee_pose, new_ee_rot
            return obs

    def step(
        self, action: Optional[np.ndarray], delta: bool = False, open_gripper: Optional[bool] = None
    ) -> Tuple[Dict[str, np.ndarray], int, bool, None]:
        """Run a step in the environment, where `delta` specifies if we are sending absolute poses or deltas in poses!"""
        if action is not None:
            if self.controller == "joint":
                if not delta:
                    # Joint Impedance Controller expects 7D Joint Angles
                    q = torch.from_numpy(action)
                    self.robot.update_desired_joint_positions(q)
                else:
                    raise NotImplementedError("Delta control for Joint Impedance Controller not yet implemented!")

            elif self.controller == "cartesian":
                if not delta:
                    # Cartesian controller expects tuple -- first 3 elements are xyz, last 4 are quaternion orientation
                    pos, quat = torch.from_numpy(action[:3]), torch.from_numpy(action[3:])
                    self.robot.update_desired_ee_pose(position=pos, orientation=quat)
                else:
                    # Convert from 6-DoF (x, y, z, roll, pitch, yaw) if necessary...
                    assert len(action) == 6, "Delta Control for Cartesian Impedance only supported for Euler Angles!"
                    pos, angle = torch.from_numpy(self.ee_position + action[:3]), self.ee_orientation + action[3:]

                    # Convert angle =>> quaternion (Polymetis expects scalar first, so roll...)
                    quat = torch.from_numpy(np.roll(R.from_euler("xyz", angle).as_quat(), 1))
                    self.robot.update_desired_ee_pose(position=pos, orientation=quat)

            elif self.controller == "resolved-rate":
                # Resolved rate controller expects 6D end-effector velocities (deltas) in X/Y/Z/Roll/Pitch/Yaw...
                ee_velocities = torch.from_numpy(action)
                self.robot.update_desired_ee_velocities(ee_velocities)

            else:
                raise NotImplementedError(f"Controller mode `{self.controller}` not supported!")

        # Gripper Handling...
        if open_gripper is not None and (self.gripper_is_open ^ open_gripper):
            # True --> Open Gripper, otherwise --> Close Gripper
            self.gripper_is_open = open_gripper
            if open_gripper:
                self.gripper.goto(GRIPPER_MAX_WIDTH, speed=GRIPPER_SPEED, force=GRIPPER_FORCE)
            else:
                self.gripper.grasp(speed=GRIPPER_SPEED, force=GRIPPER_FORCE)

        # Sleep according to control frequency
        self.rate.sleep()

        # Return observation, Gym default signature...
        return self.get_obs(), 0, False, None


    def close(self) -> None:
        # Terminate Policy
        if self.controller in {"joint", "cartesian", "resolved-rate"}:
            self.robot.terminate_current_policy()

        # Garbage collection & sleep just in case...
        del self.robot
        self.robot = None, None
        time.sleep(1)

