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

class XArmEnv(RobotEnv):
    def __init__(
    self,
    control_frequency_hz: int,
    control_mode: str = 'default',
    use_gripper: bool = False,
    use_camera: bool = False,
    use_r3m: bool = False,
    r3m_net: torch.nn.Module = None,
    xarm_ip: str = '192.168.1.220',
    random_reset_home_pose: bool = False
    ):
        self.hz = control_frequency_hz
        self.rate = Rate(control_frequency_hz)
        self.control_mode = control_mode
        self.use_gripper = use_gripper
        self.random_reset_home_pos = random_reset_home_pose
        self.xarm_ip = xarm_ip
        self.robot = None
        # TODO: This is bad, we should have a better home pos
        if self.control_mode == 'default':
            self.home_xyz = [553,29,435]
        elif self.control_mode == 'angular':
            self.home_angles = [3, 15.4, -91.8, 76.3, 4.9]
        else:
            raise NotImplementedError("This control mode not implemented yet")
        '''
        r3m useful for converting images to embeddings
        '''
        self.use_camera = use_camera
        self.use_r3m = use_r3m
        if self.use_r3m:
            self._transforms = T.Compose([
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(), # divides by 255, will also convert to chw
            ])
            if r3m_net is None:
                self.r3m = load_r3m("resnet50") # resnet18
                self.r3m.eval()
                self.r3m.to(device)
            else:
                self.r3m = r3m_net
        if self.use_camera:
            self.rgb, self.d = None, None
            self.cam = RealSenseInterface()
        if self.random_reset_home_pos:
            self._last_random_offset = np.zeros(3)
        obs = self.reset()

    def reset(self)-> Dict[str, np.ndarray]:
        self.robot_setup()
        obs = self.get_obs()
        obs = self._process_obs(obs)
        return obs

    def set_mode(self, mode:str = 'default'):
        '''
        This sets the robot into a mode to kinistetic teaching. 
        To use this, default is normal mode, and record is kinistetic mode.
        '''
        # self.mode = mode
        if mode == 'default':
            self.robot.set_mode(0)
        elif mode == 'record':
            self.robot.set_mode(2)
        self.robot.set_state(0)


    def step(
        self,
        action: Optional[np.ndarray], delta: bool = False, open_gripper: Optional[bool] = None
    ):
        '''
        Run a step in the environment, where `delta` specifies if we are sending
        absolute poses or deltas in poses!

        You can use `self.rate.sleep()` to ensure the control frequency is met
        which assumes the commands immediately return and are non-blocking
        '''

        ######
        # do something here to control robot
        ######
        if action is not None:
            # Make sure the action is an x,y,z 
            if self.control_mode == 'default':
                assert len(action) == 3
                self.move_xyz(action, deltas=delta)
            elif self.control_mode == 'angular':
                assert len(action) == 5
                self.move_angles(action, deltas=delta)
        self.rate.sleep()
        obs = self.get_obs()
        obs = self._process_obs(obs)
        reward = self._calculate_reward(obs)
        done = False
        info = {}

        return obs, reward, done, info

    def close(self):
        self.robot.disconnect()

    def _calculate_reward(self, obs):
        return 0

    def action_space(self):
        # 6-DoF (x, y, z, roll, pitch, yaw) (absolute or deltas)
        if self.control_mode == 'default':
            low = np.array([-1, -1, -1])
            hi = np.array([1, 1, 1])
            return gym.spaces.Box(low=low, high=hi, dtype=np.float32)
        elif self.control_mode == 'angular':
            low = np.array([-1, -1, -1, -1, -1])
            hi = np.array([1, 1, 1, 1, 1])
            return gym.spaces.Box(low=low, high=hi, dtype=np.float32)

    def image_space(self) -> gym.spaces.Box:
        return gym.spaces.Box(low=0, high=255, shape=(480, 640, 3), dtype=np.uint8) # HWC

    def robot_setup(self, home: str = 'default'):
        self.robot = XArmAPI(self.xarm_ip)
        self.robot.clean_warn()
        self.robot.clean_error()
        self.robot.motion_enable(enable=True)
        self.robot.set_mode(0)
        self.robot.set_state(state=0)
        print(f'Going to initial position')
        if home == 'default': 
            if self.control_mode == 'default':
                self.move_xyz(self.home_xyz, wait=True)
            elif self.control_mode == 'angular':
                self.move_angles(self.home_angles, wait=True)
        else:
            raise NotImplementedError("Only have one default hardcoded reset pos")
        if self.random_reset_home_pos:
            if self.control_mode == 'default':
                x_magnitude = 0.1
                y_magnitude = 0.25
                xyz_delta = np.array([
                    (np.random.random() * 2.0 - 1) * x_magnitude,
                    (np.random.random() * 2.0 - 1) * y_magnitude,
                    0.0
                ])
                self.move_xyz(xyz_delta, deltas=True, wait=True)
            elif self.control_mode == 'angular':
                angular_delta = np.array([(np.random.random() * 2.0 - 1) for i in range(5)])
                self.move_angles(angular_delta, deltas=True, wait=True)
        self.cur_xyz = self.get_cur_xyz()
        self.cur_angles = self.get_cur_angles()


    def get_obs(self) -> Dict[str, np.ndarray]:
        # error, new_joint_angles = self.robot.get_servo_angle()[1]
        if self.use_camera:
            self.rgb, self.d = self.cam.get_latest_rgbd()
        position = self.get_cur_xyz()
        angles = self.get_cur_angles()
        obs = {
            "q": angles,
            "delta_q": angles - self.cur_angles,
            "ee_pos": position,
            "delta_ee_pos": position - self.cur_xyz,
            
        }
        if self.use_camera:
            obs['rgb_image'] = self.rgb
            obs['d_image'] = self.d
        self.cur_xyz = position
        self.cur_angles = angles
        return obs
    

    def _process_obs(self, obs):
        if self.use_r3m:
            # convert from hwc to bchw
            pil_img = Image.fromarray(obs["rgb_image"])
            processed_image = self._transforms(pil_img).unsqueeze(0).to(device)
            with torch.no_grad():
                embedding = self.r3m(processed_image * 255.0) # r3m expects input to be 0-255
            r3m_embedding = embedding.cpu().squeeze().numpy()
            # r3m_with_ppc = np.concatenate([r3m_embedding, state_obs])
            obs["r3m_vec"] = r3m_embedding
        return obs
    
    def rgb(self) -> np.ndarray:
        '''
        return the latest rgb image
        (useful if you need to query for the last image outside of an RL loop)
        '''
        assert self.use_camera
        return self.rgb

    def move_xyz(self, xyz:np.ndarray, deltas: bool = False, wait: bool = False) -> None:
        if deltas:
            # We might not want to get the cur xyz here and instead use the self.curxyz
            cur_pos = self.cur_xyz
            xyz = np.add(cur_pos, xyz)
        self.robot.set_position(x=xyz[0], y= xyz[1], z=xyz[2], wait=wait)
    
    def get_cur_xyz(self) -> np.ndarray:
        error, position = self.robot.get_position()
        if error != 0:
            raise NotImplementedError('Need to handle xarm exception')
        return np.array(position[:3])
    
    def move_angles(self, angles: np.ndarray, deltas: bool = False, wait: bool=False) -> None:
        if deltas:
            angles = np.add(angles, self.cur_angles)
        # Have to add on two zeros to the angles
        angles = np.pad(angles, (0,2), 'constant')
        self.robot.set_servo_angle(angle=angles, wait=wait)

    def get_cur_angles(self) -> np.ndarray:
        error, angles = self.robot.get_servo_angle()
        if error != 0:
            raise NotImplementedError('Need to handle angle error')
        return np.array(angles[:5])
    
    def close(self):
        if self.control_mode == 'default':
            self.move_xyz(self.home_xyz, wait=True)
        elif self.control_mode == 'angular':
            self.move_angles(self.home_angles, wait=True)
        self.robot.disconnect()
