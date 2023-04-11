import time
from robot.base_robot import RobotEnv
from typing import Any, Dict, List, Optional, Tuple
from cam.realsense import RealSenseInterface
from PIL import Image
import gym
import numpy as np
import torch
import torchvision.transforms as T
from gym.spaces import Dict as GymDict
# export XLA_PYTHON_CLIENT_MEM_FRACTION=0.7
XARM_SDK = '/home/xarm/Desktop/xArm-Python-SDK'
import sys

sys.path.append(XARM_SDK)
from xarm.wrapper import XArmAPI
from reward_extraction.reward_functions import RobotLearnedRewardFunction
from robot.utils import Rate
from r3m import load_r3m
import rospy
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from std_msgs.msg import String, Float32MultiArray


class XArmCmSafeEnvironment(RobotEnv):
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
            low_collision_sensitivity: bool = False,
        ):
        # Properties that are needed, but we don't use ourselves
        self.spec = None
        self.hz = control_frequency_hz

        rospy.init_node('robot_node')
        pub = rospy.Publisher('commanded_positions', Float32MultiArray)
        self.pub = pub

        self.min_box = [26.0, -23.1, 13]
        self.max_box = [54.4, 20.7, 44.0]
        
        self.rate = Rate(control_frequency_hz)
        self.use_gripper = use_gripper
        self.random_reset_home_pos = random_reset_home_pose
        self.xarm_ip = xarm_ip
        self.collision_sensitivity = low_collision_sensitivity
        self.scale_factor = scale_factor
        self.robot = XArmAPI(self.xarm_ip); self.robot.connect(port=self.xarm_ip)
        self.movement_mode()
        self.robot.set_vacuum_gripper(False)
        self.r = rospy.Rate(control_frequency_hz)
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
                self.r3m = load_r3m("resnet18")  # resnet18
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

    def movement_mode(self):
        self.robot.motion_enable(enable=True)
        self.robot.set_mode(1)
        self.robot.set_state(0)

    def step(
            self,
            action: Optional[np.ndarray], delta: bool = True):
        """
        Run a step in the environment, where `delta` specifies if we are sending
        absolute poses or deltas in poses!

        You can use `self.rate.sleep()` to ensure the control frequency is met
        which assumes the commands immediately return and are non-blocking
        """
        ######
        # do something here to control robot
        ######
        _, codes = self.robot.get_err_warn_code()
        error_code = codes[0]
        if error_code != 0:
            self.robot.clean_error()
            self.movement_mode()
        if action is not None:
            # Make sure the action is an x,y,z 
            if self.use_gripper:
                assert action.shape[0] == 4
                self.update_gripper(action[3])
            else:
                assert action.shape[0] == 3
            self.move_xyz(action[:3], deltas=delta)
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

    def _calculate_reward(self, obs):
        return 0

    @property
    def action_space(self):
        # 3DOF (x,y,z) (absolute or deltas) and gripper potentially
        if self.use_gripper:
            low = np.array([-1, -1, -1, -1])
            hi = np.array([1, 1, 1, 1])
        else:
            low = np.array([-1, -1, -1])
            hi = np.array([1, 1, 1])
        return gym.spaces.Box(low=low, high=hi, dtype=np.float32)

    @property
    def image_space(self) -> gym.spaces.Box:
        return gym.spaces.Box(low=0, high=255, shape=(480, 640, 3), dtype=np.uint8)  # HWC

    def robot_setup(self, home: str = 'default'):
        self.gripper_val = -1
        if self.collision_sensitivity:
            self.robot.set_collision_sensitivity(0)
        print(f'Going to initial position')
        if home == 'default':
            self.movement_mode()
            # In order to fix Kinematics error, if you just force reset to jas, it works
            # self.robot.set_servo_angle(angle=[3.000007, 15.400017, -91.799985, 76.399969, 4.899992, 0.0, 0.0],
            #                            wait=True)
            self.move_xyz(np.array([55.3490479, 2.9007273, 42.4868439]), wait=True)
            time.sleep(2)
            self.robot.set_vacuum_gripper(False)
        else:
            raise NotImplementedError("Only have one default hardcoded reset pos")
        if self.random_reset_home_pos:
            x_magnitude = 0.1
            y_magnitude = 0.25
            xyz_delta = np.array([
                (np.random.random() * 2.0 - 1) * x_magnitude,
                (np.random.random() * 2.0 - 1) * y_magnitude,
                0.0
            ])
            self.robot.move_xyz(xyz_delta, deltas=True, wait=True)
            time.sleep(1)
        self.cur_xyz = self.get_cur_xyz()

    def get_obs(self) -> Dict[str, np.ndarray]:
        if self.use_camera:
            self.rgb, self.d = self.cam.get_latest_rgbd()
        position = self.get_cur_xyz()

        obs = {
            "ee_pos": position,
            "delta_ee_pos": (position - self.get_cur_xyz()) / self.scale_factor,
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
            xyz = xyz.clip(-1, 1)
            xyz *= self.scale_factor
            cur_pos = self.get_cur_xyz()
            xyz = np.add(cur_pos, xyz)
        # Clamp it to be within the min and max poses. The min and max are in centimeters
        # and clamped to centimeters
        for i in range(len(xyz)):
            xyz[i] = max(self.min_box[i], xyz[i])
            xyz[i] = min(self.max_box[i], xyz[i])
    
        self.movement_mode()

        message = Float32MultiArray()
        # Xarm takes mm, but xyz is in cm
        message.data = np.asarray(xyz*10).copy()
        self.pub.publish(message)

        self.r.sleep()
        if wait:
            self.wait_until_stopped()
    def get_cur_xyz(self) -> np.ndarray:
        # In cm
        error, position = self.robot.get_position()
        if error != 0:
            raise NotImplementedError('Need to handle xarm exception')
        # position is in mm, want to return it in cm so devide by 10
        return np.array(position[:3]) / 10

    def wait_until_stopped(self):
        while True:
            time.sleep(.1)
            if not self.robot.get_is_moving():
                break

class RLReadyXarmEnvironment(XArmCmSafeEnvironment):
    def _process_obs(self, obs):
        new_obs = {}
        # print(obs)
        state_obs = obs['ee_pos']
        new_obs['ee_pos'] = state_obs
        new_obs['obs'] = state_obs

        new_obs['delta_ee_pos'] = obs['delta_ee_pos']
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

    @property
    def observation_space(self):
        obs_dict = GymDict()
        base_obs_dim = 3  # eef pose (xyz position)
        if self.use_gripper:
            # If gripper, we have one more
            base_obs_dim += 1
        obs_dict['ee_pos'] = gym.spaces.Box(low=float("-inf"), high=float("inf"), shape=(base_obs_dim,), dtype=np.float32)
        obs_dict["obs"] = gym.spaces.Box(low=float("-inf"), high=float("inf"), shape=(base_obs_dim,), dtype=np.float32)
        obs_dict["delta_ee_pos"] = gym.spaces.Box(low=float("-inf"), high=float("inf"), shape=(base_obs_dim,), dtype=np.float32)

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
    

class SimpleRealXArmReach(RLReadyXarmEnvironment):
    def __init__(self, goal, **kwargs):
        super().__init__(**kwargs)
        self.goal = goal

    def _calculate_reward(self, obs):
        # Current goal position is '[60.4, -5.47, 35.41]'
        dist = np.linalg.norm(obs['obs'][:3] - self.goal)
        return -dist