import time
from robot.base_robot import RobotEnv
from typing import Any, Dict, List, Optional, Tuple
from cam.realsense import RealSenseInterface
from PIL import Image
import gym
import numpy as np
import torch
import torchvision.transforms as T

XARM_SDK = '/home/xarm/Desktop/xArm-Python-SDK'
import sys

sys.path.append(XARM_SDK)
from xarm.wrapper import XArmAPI
from reward_extraction.reward_functions import RobotLearnedRewardFunction
from robot.utils import Rate
from r3m import load_r3m

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class XArmBaseEnvironment(RobotEnv):
    def __init__(
            self,
            control_frequency_hz: int,
            scale_factor: float = 10,
            use_gripper: bool = False,
            use_camera: bool = False,
            use_r3m: bool = False,
            r3m_net: torch.nn.Module = None,
            xarm_ip: str = '192.168.1.220',
            random_reset_home_pose: bool = False,
            speed: float = 1,
            low_collision_sensitivity: bool = False,
            # noisy: bool = False
    ):
        self.hz = control_frequency_hz
        self.rate = Rate(control_frequency_hz)
        self.use_gripper = use_gripper
        self.random_reset_home_pos = random_reset_home_pose
        self.xarm_ip = xarm_ip
        self.robot = None
        self.mode = 'default'
        self.speed = speed
        self.collision_sensitivity = low_collision_sensitivity
        self.scale_factor = scale_factor
        # self.noisy = noisy
        '''
        r3m useful for converting images to embeddings
        '''
        self.use_camera = use_camera
        self.use_r3m = use_r3m
        if self.use_r3m:
            self._transforms = T.Compose([
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),  # divides by 255, will also convert to chw
            ])
            if r3m_net is None:
                self.r3m = load_r3m("resnet50")  # resnet18
                self.r3m.eval()
                self.r3m.to(device)
            else:
                self.r3m = r3m_net
        if self.use_camera:
            self.rgb, self.d = None, None
            self.cam = RealSenseInterface()
        obs = self.reset()

    def reset(self) -> Dict[str, np.ndarray]:
        self.robot_setup()
        obs = self.get_obs()
        obs = self._process_obs(obs)
        return obs

    def set_mode(self, mode: str = 'default'):
        '''
        This sets the robot into a mode to kinistetic teaching. 
        To use this, default is normal mode, and record is kinistetic mode.
        '''
        self.mode = mode
        if mode == 'default':
            self.robot.set_mode(7)
        elif mode == 'record':
            self.robot.set_mode(2)
        self.robot.set_state(0)

    def step(
            self,
            action: Optional[np.ndarray], delta: bool = False):
        """
        Run a step in the environment, where `delta` specifies if we are sending
        absolute poses or deltas in poses!

        You can use `self.rate.sleep()` to ensure the control frequency is met
        which assumes the commands immediately return and are non-blocking
        """

        ######
        # do something here to control robot
        ######
        if action is not None:
            # Make sure the action is an x,y,z 
            if self.use_gripper:
                assert action.shape[0] == 4
                self.update_gripper(action[3])
            else:
                assert action.shape[0] == 3
            self.move_xyz(action[:3], deltas=delta)

        self.rate.sleep()
        obs = self.get_obs()
        obs = self._process_obs(obs)
        reward = self._calculate_reward(obs)
        done = False
        info = {}

        return obs, reward, done, info

    def update_gripper(self, command: float):
        if command > 0:
            self.gripper_val = 1
            self.robot.set_vacuum_gripper(True)
        else:
            self.gripper_val = -1
            self.robot.set_vacuum_gripper(False)

    def close(self):
        self.robot.disconnect()

    # def _calculate_reward(self, obs):
    #     return 0

    def action_space(self):
        # 6-DoF (x, y, z, roll, pitch, yaw) (absolute or deltas)
        if self.use_gripper:
            low = np.array([-1, -1, -1, -1])
            hi = np.array([1, 1, 1, 1])
        else:
            low = np.array([-1, -1, -1])
            hi = np.array([1, 1, 1])
        return gym.spaces.Box(low=low, high=hi, dtype=np.float32)

    def image_space(self) -> gym.spaces.Box:
        return gym.spaces.Box(low=0, high=255, shape=(480, 640, 3), dtype=np.uint8)  # HWC

    def robot_setup(self, home: str = 'default'):
        self.robot = XArmAPI(self.xarm_ip)
        self.robot.motion_enable(enable=True)
        self.robot.set_mode(7)
        self.robot.set_state(state=0)
        self.robot.set_vacuum_gripper(False)
        self.gripper_val = -1
        if self.collision_sensitivity:
            self.robot.set_collision_sensitivity(0)
        # self.robot.set_tcp_load(0.2, [0, 0, 0])
        print(f'Going to initial position')
        if home == 'default':
            self.robot.set_mode(0)
            self.robot.set_state(0)
            # In order to fix Kinematics error, if you just force reset to jas, it works
            self.robot.set_servo_angle(angle=[3.000007, 15.400017, -91.799985, 76.399969, 4.899992, 0.0, 0.0],
                                       wait=True)
            self.robot.set_mode(7)
            self.robot.set_state(0)
            # self.robot.set_tcp_load(0.2, [0, 0, 0])
        else:
            raise NotImplementedError("Only have one default hardcoded reset pos")
        if self.random_reset_home_pos:
            x_magnitude = 0.1
            y_magnitude = 0.25
            # TODO: scale this from -1 to 1
            xyz_delta = np.array([
                (np.random.random() * 2.0 - 1) * x_magnitude,
                (np.random.random() * 2.0 - 1) * y_magnitude,
                0.0
            ])
            self.robot.move_xyz(xyz_delta, deltas=True, wait=True)
        self.cur_xyz = self.get_cur_xyz()

    def get_obs(self) -> Dict[str, np.ndarray]:
        # error, new_joint_angles = self.robot.get_servo_angle()[1]
        if self.use_camera:
            self.rgb, self.d = self.cam.get_latest_rgbd()
        position = self.get_cur_xyz()

        # This is not how we should add noise
        # if self.noisy: 
        #     position += np.random.normal(0, 0.05, position.shape)

        obs = {
            # TODO: Change this for the gripper
            "ee_pos": position,
            "delta_ee_pos": (position - self.cur_xyz) / self.scale_factor,
        }

        if self.use_gripper:
            obs['ee_pos'] = np.append(obs['ee_pos'], self.gripper_val)
            obs['delta_ee_pos'] = np.append(obs['delta_ee_pos'], self.gripper_val)

        if self.use_camera:
            obs['rgb_image'] = self.rgb
            obs['d_image'] = self.d
        self.cur_xyz = position
        return obs

    def _process_obs(self, obs):
        # For overiding
        return obs

    def rgb(self) -> np.ndarray:
        '''
        return the latest rgb image
        (useful if you need to query for the last image outside of an RL loop)
        '''
        assert self.use_camera
        return self.rgb

    def move_xyz(self, xyz: np.ndarray, deltas: bool = False, wait: bool = False) -> None:
        if deltas:
            # # TODO: Make this actually clip it and use a paramater, not 6
            xyz = xyz.clip(-1, 1)
            xyz *= self.scale_factor
            # We might not want to get the cur xyz here and instead use the self.curxyz
            cur_pos = self.cur_xyz
            xyz = np.add(cur_pos, xyz)
        self.robot.set_position(x=xyz[0], y=xyz[1], z=xyz[2], speed=self.speed)
        # Janky wait code.

        if wait:
            self.wait_until_stopped()

    def get_cur_xyz(self) -> np.ndarray:
        error, position = self.robot.get_position()
        if error != 0:
            raise NotImplementedError('Need to handle xarm exception')
        return np.array(position[:3])

    def wait_until_stopped(self):
        while True:
            time.sleep(.1)
            if not self.robot.get_is_moving():
                break


# This converts the unites of end effector positons into centemeters. This means that it returns centimeeters
# and is passed centimeters.
class XArmCentimeterBaseEnviornment(XArmBaseEnvironment):
    def move_xyz(self, xyz: np.ndarray, deltas: bool = False, wait: bool = False) -> None:
        if deltas:
            # We might not want to get the cur xyz here and instead use the self.curxyz
            xyz = xyz.clip(-1, 1)
            xyz *= self.scale_factor
            cur_pos = self.cur_xyz
            xyz = np.add(cur_pos, xyz)
        self.robot.set_position(x=xyz[0] * 10, y=xyz[1] * 10, z=xyz[2] * 10, speed=self.speed)

        if wait:
            self.wait_until_stopped()

    def get_cur_xyz(self) -> np.ndarray:
        return super().get_cur_xyz() / 10


class XArmCentimeterSafeEnvironment(XArmCentimeterBaseEnviornment):
    """
    This arm will not go outside the safty box that we have constructed.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.min_box = [17.9, -45.8, 14]
        self.max_box = [69, 45, 62]

    def move_xyz(self, xyz: np.ndarray, deltas: bool = False, wait: bool = False) -> None:
        if deltas:
            xyz = xyz.clip(-1, 1)
            xyz *= self.scale_factor
            cur_pos = self.cur_xyz
            xyz = np.add(cur_pos, xyz)
        # Clamp it to be within the min and max poses. The min and max are in centimeters
        # and clamped to centimeters
        for i in range(len(xyz)):
            xyz[i] = max(self.min_box[i], xyz[i])
            xyz[i] = min(self.max_box[i], xyz[i])

        self.robot.set_position(x=xyz[0] * 10, y=xyz[1] * 10, z=xyz[2] * 10, speed=self.speed)
        if wait:
            self.wait_until_stopped()

    def _process_obs(self, obs):
        new_obs = {}
        state_obs = obs['ee_pose']
        new_obs['obs'] = state_obs
        new_obs['eef_deltas'] = obs['eef_deltas']
        if self.use_camera:
            new_obs["rgb_image"] = obs["rgb_image"]
            if self.use_r3m:
                # convert from hwc to bchw
                pil_img = Image.fromarray(obs["rgb_image"])
                processed_image = self._transforms(pil_img).unsqueeze(0).to(device)
                with torch.no_grad():
                    embedding = self.r3m(processed_image * 255.0)  # r3m expects input to be 0-255
                r3m_embedding = embedding.cpu().squeeze().numpy()
                r3m_with_ppc = np.concatenate([r3m_embedding, state_obs])
                new_obs["r3m_vec"] = r3m_embedding
                new_obs["r3m_with_ppc"] = r3m_with_ppc

        return new_obs

    def observation_space(self):
        obs_dict = GymDict()
        base_obs_dim = 3  # eef pose (xyz position)
        if self.use_gripper:
            # If gripper, we have one more
            base_obs_dim += 1

        obs_dict["obs"] = gym.spaces.Box(low=float("-inf"), high=float("inf"), shape=(base_obs_dim,), dtype=np.float32)
        obs_dict["eef_deltas"] = gym.spaces.Box(low=float("-inf"), high=float("inf"), shape=(base_obs_dim,), dtype=np.float32)

        if self.use_camera:
            obs_dict["rgb_image"] = gym.spaces.Box(low=0, high=255, shape=(480, 640, 3), dtype=np.uint8)  # HWC
            if self.use_r3m:
                # resnet18 - 512, resnet50 - 2048
                r3m_embedding_dim = 512
                obs_dict["r3m_vec"] = gym.spaces.Box(low=float("-inf"), high=float("inf"), shape=(r3m_embedding_dim,),
                                                     dtype=np.float32)
                obs_dict["r3m_with_ppc"] = gym.spaces.Box(low=float("-inf"), high=float("inf"),
                                                          shape=(base_obs_dim + r3m_embedding_dim,), dtype=np.float32)

        return obs_dict


class SimpleRealXArmReach(XArmCentimeterSafeEnvironment):
    def __init__(self, goal, **kwargs):
        self.goal = goal
        XArmBaseEnvironment.__init__(self, **kwargs)

    def _calculate_reward(self, obs, reward, info):
        # Current goal position is '[60.4, -5.47, 35.41]'
        dist = np.linalg.norm(obs['obs'][:3] - self.goal)
        return -dist


