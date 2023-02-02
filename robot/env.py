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
from gym.spaces import Dict as GymDict
from polymetis import GripperInterface, RobotInterface
from PIL import Image
from scipy.spatial.transform import Rotation as R
import torchvision.transforms as T

from cam.realsense import RealSenseInterface
from robot.controllers import ResolvedRateControl
from robot.utils import (
    ROBOT_IP, HOMES, HZ,
    KQ_GAINS, KQD_GAINS, KX_GAINS, KXD_GAINS, KRR_GAINS,
    GRIPPER_MAX_WIDTH, GRIPPER_FORCE, GRIPPER_SPEED,
    Rate
)
from r3m import load_r3m

# Silence OpenAI Gym warnings
gym.logger.setLevel(logging.ERROR)

def normalize_gripper(val):
    # normalize gripper such that 1 is fully open and 0 is closed
    normalized = val / GRIPPER_MAX_WIDTH
    return max(min(normalized, 1.0), 0.0)

"""
wrapper that addds the resolved rate controller + changes quaternion order to xyzw
"""
class RobotInterfaceWrapper(RobotInterface):
    def get_ee_pose_with_rot_as_xyzw(self) -> Tuple[torch.Tensor, torch.Tensor]:
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
        cam_serial_str: str = None,
        use_gripper: bool = False,
        random_reset_home_pose: bool = False,
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
        self.hz = hz
        self.rate = Rate(hz)
        self.mode = mode
        self.controller = controller
        self.curr_step = 0

        self.current_joint_pose, self.current_ee_pose, self.current_ee_rot = None, None, None
        self.robot, self.kp, self.kpd = None, None, None
        self.use_gripper, self.gripper, self.current_gripper_state, self.gripper_is_open = use_gripper, None, None, True

        self.use_camera = use_camera
        if self.use_camera:
            self.rgb, self.d = None, None
            self.cam = RealSenseInterface()

        # random offset from "home" pose
        self.random_reset_home_pose = random_reset_home_pose
        self._last_random_offset = np.zeros(3)

        # Initialize Robot and PD Controller
        obs = self.reset()

        '''
        q: (7,) (float32)
        qdot: (7,) (float32)
        delta_q: (7,) (float32)
        ee_pose: (7,) (float32)
        delta_ee_pose: (7,) (float32)
        gripper_width: 0.07800412178039551 (<class 'float'>)      # use_gripper = True
        gripper_max_width: 0.07867326587438583 (<class 'float'>)  # use_gripper = True
        gripper_open: True (<class 'bool'>)                       # use_gripper = True
        rgb_image: (480, 640, 3) (uint8)                          # use_camera = True
        d_image: (480, 640) (uint16)                              # use_camera = True
        '''
        print(f"---base robot env observation space")
        for k, v in obs.items():
            if type(v) == np.ndarray:
                print(f"{k}: {v.shape} ({v.dtype})")
            else:
                print(f"{k}: {v} ({type(v)})")

    def robot_setup(self, home: str = "default", franka_ip: str = ROBOT_IP, use_last_reset_pos: bool = False) -> None:
        # Initialize Robot Interface and Reset to Home
        self.robot = RobotInterfaceWrapper(ip_address=franka_ip)
        self.robot.set_home_pose(torch.Tensor(HOMES[home]))
        print(f"Robot going home!")
        self.robot.go_home()

        if self.random_reset_home_pose:
            if use_last_reset_pos:
                print(self._last_random_offset)
                xyz_delta = self._last_random_offset
            else:
                pos_xyz, rot_xyzw = self.robot.get_ee_pose_with_rot_as_xyzw()
                x_magnitude = 0.1
                y_magnitude = 0.25
                xyz_delta = torch.Tensor([
                    (np.random.random() * 2.0 - 1) * x_magnitude,
                    (np.random.random() * 2.0 - 1) * y_magnitude,
                    0.0
                ])
                # xyz_delta = np.zeros(3)
                self._last_random_offset = torch.clone(xyz_delta)
            print(f"Robot going to delta {xyz_delta}, flag: {use_last_reset_pos}!")
            self.robot.move_to_ee_pose(position=xyz_delta, delta=True)

        # Initialize current joint & EE poses...
        self.current_ee_pose = np.concatenate([a.numpy() for a in self.robot.get_ee_pose_with_rot_as_xyzw()])
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

    def _process_obs(self, obs):
        # child classes could do something fancier with the obs dict
        return obs

    def reset(self, use_last_reset_pos: bool = False) -> Dict[str, np.ndarray]:
        # Set PD Gains -- kp, kpd -- depending on current mode, controller
        if self.controller == "joint":
            self.kp, self.kpd = KQ_GAINS[self.mode], KQD_GAINS[self.mode]
        elif self.controller == "cartesian":
            self.kp, self.kpd = KX_GAINS[self.mode], KXD_GAINS[self.mode]
        elif self.controller == "resolved-rate":
            self.kp = KRR_GAINS[self.mode]

        # Call setup with the new controller...
        self.robot_setup(use_last_reset_pos=use_last_reset_pos)

        obs = self.get_obs()
        return self._process_obs(obs)

    def set_mode(self, mode: str) -> None:
        self.mode = mode

    def get_obs(self) -> Dict[str, np.ndarray]:
        new_joint_pose = self.robot.get_joint_positions().numpy()
        new_ee_pose = np.concatenate([a.numpy() for a in self.robot.get_ee_pose_with_rot_as_xyzw()])
        new_ee_rot = R.from_quat(new_ee_pose[3:]).as_euler("xyz")

        if self.use_camera:
            self.rgb, self.d = self.cam.get_latest_rgbd()

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
            if self.use_camera:
                obs["rgb_image"] = self.rgb
                obs["d_image"] = self.d

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
            if self.use_camera:
                obs["rgb_image"] = self.rgb
                obs["d_image"] = self.d

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
        obs = self.get_obs()
        return self._process_obs(obs), 0, False, {}


    def close(self) -> None:
        # Terminate Policy
        if self.controller in {"joint", "cartesian", "resolved-rate"}:
            self.robot.terminate_current_policy()

        # Garbage collection & sleep just in case...
        del self.robot
        self.robot = None

        if self.use_gripper:
            del self.gripper
            self.gripper = None

        # note we don't kill the camera here since we just want it to stay on

        time.sleep(1)

    @property
    def ee_position(self) -> np.ndarray:
        """Return current EE position --> 3D x/y/z!"""
        return self.current_ee_pose[:3]

    @property
    def ee_orientation(self) -> np.ndarray:
        """Return current EE orientation as euler angles (in radians) --> 3D roll/pitch/yaw!"""
        return self.current_ee_rot

'''
empirically moved around the arm to figure out where it could reach without probably running into joint lock
'''
FRANKA_XYZ_MIN = np.array([
    0.28, -0.25, 0.16
])
FRANKA_XYZ_MAX = np.array([
    0.7, 0.25, 0.58
])

class SafeTaskSpaceFrankEnv(FrankaEnv):
    '''
    Real franka env that also sandboxes the robot into a safe workspace.

    real robot tasks should inherit from this class and redefine how reward is calculated
    '''
    def __init__(
        self,
        xyz_min: np.ndarray = FRANKA_XYZ_MIN,
        xyz_max: np.ndarray = FRANKA_XYZ_MAX,
        use_r3m: bool = False,
        only_pos_control: bool = True,
        **kwargs
    ):
        self.xyz_min = xyz_min
        self.xyz_max = xyz_max
        self.action_scale = 0.1
        self.use_r3m = use_r3m
        self.only_pos_control = only_pos_control
        if self.use_r3m:
            self.r3m = load_r3m("resnet50") # resnet18
            self._transforms = T.Compose([
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(), # divides by 255, will also convert to chw
            ])
            self.torch_device = "cuda"
            self.r3m.eval()
            self.r3m.to(self.torch_device)

        FrankaEnv.__init__(self, **kwargs)
        assert self.controller == "cartesian"

    @property
    def action_space(self):
        if self.controller == "joint":
            # 7D Joint Angles
            low = np.array([-np.pi for i in range(7)])
            hi = np.array([np.pi for i in range(7)])
        elif self.controller == "cartesian":
            # 6-DoF (x, y, z, roll, pitch, yaw) (absolute or deltas)
            low = np.array([-1, -1, -1])
            hi = np.array([1, 1, 1])
            if not self.only_pos_control:
                low = np.append(low, [-np.pi, -np.pi, -np.pi])
                hi = np.append(hi, [np.pi, np.pi, np.pi])
        elif self.controller == "resolved-rate":
            # 6D end-effector velocities (deltas) in X/Y/Z/Roll/Pitch/Yaw...
            low = np.array([float("-inf") for i in range(6)])
            hi = np.array([float("inf") for i in range(6)])
        else:
            raise NotImplementedError(f"Controller mode `{self.controller}` not supported!")

        # let's say gripper open is positive vals and gripper closed is negative vals
        if self.use_gripper:
            np.append(low, -1.0)
            np.append(hi, 1.0)

        return gym.spaces.Box(low=low, high=hi, dtype=np.float32)

    @property
    def observation_space(self):
        obs_dict = GymDict()
        base_obs_dim = 14 # eef pose (xyz position xyzw rotation, joint angles)
        if self.use_gripper:
            base_obs_dim += 1

        obs_dict["obs"] = gym.spaces.Box(low=float("-inf"), high=float("inf"), shape=(base_obs_dim,), dtype=np.float32)

        if self.use_camera:
            obs_dict["rgb_image"] = gym.spaces.Box(low=0, high=255, shape=(480, 640, 3), dtype=np.uint8) # HWC
            if self.use_r3m:
                # resnet18 - 512, resnet50 - 2048
                r3m_embedding_dim = 2048
                obs_dict["r3m_vec"] = gym.spaces.Box(low=float("-inf"), high=float("inf"), shape=(r3m_embedding_dim,), dtype=np.float32)
                obs_dict["r3m_with_ppc"] = gym.spaces.Box(low=float("-inf"), high=float("inf"), shape=(base_obs_dim + r3m_embedding_dim,), dtype=np.float32)

        return obs_dict

    def _process_obs(self, obs):
        new_obs = {}
        state_obs = np.concatenate([obs["ee_pose"], obs["q"]])

        if self.use_gripper:
            gripper_val = normalize_gripper(obs["gripper_width"])
            state_obs = np.append(state_obs, gripper_val)
        new_obs["obs"] = state_obs

        if self.use_camera:
            new_obs["rgb_image"] = obs["rgb_image"]
            if self.use_r3m:
                # convert from hwc to bchw
                pil_img = Image.fromarray(obs["rgb_image"])
                processed_image = self._transforms(pil_img).unsqueeze(0).to(self.torch_device)
                with torch.no_grad():
                    embedding = self.r3m(processed_image * 255.0) # r3m expects input to be 0-255
                r3m_embedding = embedding.cpu().squeeze().numpy()
                r3m_with_ppc = np.concatenate([r3m_embedding, state_obs])
                new_obs["r3m_vec"] = r3m_embedding
                new_obs["r3m_with_ppc"] = r3m_with_ppc

        return new_obs

    def _calculate_reward(self, obs, reward, info):
        '''
        note that this will take the processed obs, not the raw obs!
        '''
        return reward

    def step(
        self, action: Optional[np.ndarray], delta: bool = True
    ) -> Tuple[Dict[str, np.ndarray], int, bool, None]:
        curr_ee_xyz = self.current_ee_pose[:3]

        gripper_action = 1.0
        if self.use_gripper:
            gripper_action = action[-1]

        if self.controller == "cartesian":
            if delta:
                # actions are coming in between -1, 1 (probably?)
                scaled_xyz_action = action[:3] * self.action_scale

                # delta should keep us within the sandbox
                max_delta = self.xyz_max - curr_ee_xyz
                min_delta = self.xyz_min - curr_ee_xyz

                clamped_action = np.array([
                    max(min(max_delta[0], scaled_xyz_action[0]), min_delta[0]),
                    max(min(max_delta[1], scaled_xyz_action[1]), min_delta[1]),
                    max(min(max_delta[2], scaled_xyz_action[2]), min_delta[2]),
                ])

                if self.only_pos_control:
                    clamped_action = np.append(clamped_action, [0.0, 0.0, 0.0])
                else:
                    clamped_action = np.append(clamped_action, [action[3], action[4], action[5]])
            else:
                assert False # this seems.... dangerous

                # absolute position targets should remain within the sandbox
                clamped_action = np.array([
                    np.max(np.min(self.xyz_max[0], action[0]), self.xyz_max[0]),
                    np.max(np.min(self.xyz_max[1], action[1]), self.xyz_max[1]),
                    np.max(np.min(self.xyz_max[2], action[2]), self.xyz_max[2]),
                    action[3], action[4], action[5],
                ])
        else:
            raise NotImplementedError(f"Havevn't thought through space clamping for this controller!")

        obs, reward, done, info = super().step(action=clamped_action, delta=delta, open_gripper=(gripper_action >= 0.0))

        reward = self._calculate_reward(obs, reward, info)

        return obs, reward, done, info


class SimpleRealFrankReach(SafeTaskSpaceFrankEnv):
    def __init__(self, goal, **kwargs):
        self.goal = goal
        SafeTaskSpaceFrankEnv.__init__(self, **kwargs)

    def _calculate_reward(self, obs, reward, info):
        dist = np.linalg.norm(obs['obs'][:3] - self.goal)
        return -dist

class LrfRealFrankaReach(SafeTaskSpaceFrankEnv):
    from reward_extraction.reward_functions import RobotLearnedRewardFunction

    def __init__(self, **kwargs):
        self.lrf = None
        SafeTaskSpaceFrankEnv.__init__(self, **kwargs)

    def set_lrf(self, lrf: RobotLearnedRewardFunction):
        print(f"learned reward function set!")
        self.lrf = lrf

    def _calculate_reward(self, obs, reward, info):
        assert self.lrf is not None

        state = torch.from_numpy(obs["r3m_vec"]).float()
        reward = self.lrf._calculate_reward(state)
        return reward


if __name__ == "__main__":
    # baseline code to make sure env is working and debug
    env = FrankaEnv(
        home="default",
        hz=HZ,
        controller="cartesian",
        mode="default",
        use_camera=True,
        use_gripper=True,
    )
    env.close()