from abc import ABC, abstractmethod
import random
from typing import Any, Dict, List, Optional, Tuple

import gym
import numpy as np
import torch
import torchvision.transforms as T

from reward_extraction.reward_functions import RobotLearnedRewardFunction
from robot.utils import Rate
from r3m import load_r3m

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class RobotEnv:
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def step(self, action):
        pass

    @abstractmethod
    def reset(self):
        pass

    @property
    @abstractmethod
    def action_space(self):
        pass

    @property
    @abstractmethod
    def observation_space(self):
        pass

class TaskSpaceRobotEnv(RobotEnv):
    '''
    Robot environment with a camera where the control is in task space
    '''
    def __init__(
        self,
        control_frequency_hz: int,
        only_pos_control: bool = True,
        use_gripper: bool = False,
        use_camera: bool = False,
        use_r3m: bool = False,
        r3m_net: torch.nn.Module = None,
    ):
        self.hz = control_frequency_hz
        self.rate = Rate(control_frequency_hz)
        self.only_pos_control = only_pos_control
        self.use_gripper = use_gripper

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

        self.rate.sleep()
        obs = self._get_obs()
        reward = self._calculate_reward(obs)
        done = False
        info = {}

        return obs, reward, done, info

    @property
    def action_space(self):
        # 6-DoF (x, y, z, roll, pitch, yaw) (absolute or deltas)
        low = np.array([-1, -1, -1])
        hi = np.array([1, 1, 1])
        if not self.only_pos_control:
            low = np.append(low, [-np.pi, -np.pi, -np.pi])
            hi = np.append(hi, [np.pi, np.pi, np.pi])

        # let's say gripper open is positive vals and gripper closed is negative vals
        if self.use_gripper:
            np.append(low, -1.0)
            np.append(hi, 1.0)

        return gym.spaces.Box(low=low, high=hi, dtype=np.float32)

    @property
    def image_space(self) -> gym.spaces.Box:
        return gym.spaces.Box(low=0, high=255, shape=(480, 640, 3), dtype=np.uint8) # HWC

    @property
    def rgb(self) -> np.ndarray:
        '''
        return the latest rgb image
        (useful if you need to query for the last image outside of an RL loop)
        '''
        assert self.use_camera
        return self._rgb

    def _get_obs(self) -> Dict[str, np.ndarray]:
        '''
        get the latest camera image
        get the latest robot state (joint angles, eef, gripper state, etc.)
        '''
        obs_dict = {}

        if self.use_camera:
            self._rgb = None

            if self.use_r3m:
                '''
                note that r3m expects input to be 0-255 instead of 0-1!
                '''
                obs_dict["r3m_vec"] = None

        return obs_dict

    def _calculate_reward(self, obs) -> float:
        '''
        calculate reward given current state
        (process and pass to a learned function, calculate some l2 distance, etc.)
        '''
        return 0

class LrfTaskSpaceRobotEnv(TaskSpaceRobotEnv):
    def set_lrf(self, lrf: RobotLearnedRewardFunction):
        self.lrf = lrf

    def _calculate_reward(self, obs):
        assert self.lrf is not None
        state = torch.from_numpy(obs["r3m_vec"]).float()
        reward = self.lrf._calculate_reward(state)
        return reward
