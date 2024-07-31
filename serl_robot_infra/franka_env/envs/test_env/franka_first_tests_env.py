import copy
import numpy as np
import gym
import time



from franka_env.envs.franka_env import FrankaEnv
from franka_env.utils.rotations import euler_2_quat
from franka_env.envs.test_env.config import TestEnvConfig


class FrankaTest(FrankaEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, config=TestEnvConfig)
    
  





    def go_to_rest(self, joint_reset=False):

        self._send_gripper_command(-1)
        self._update_currpos()
        self._send_pos_command(self.currpos)
        time.sleep(0.5)

        # Move up to clear the slot
        self._update_currpos()
        reset_pose = copy.deepcopy(self.currpos)
        reset_pose[2] += 0.10
        self.interpolate_move(reset_pose, timeout=1)
        # execute the go_to_rest method from the parent class
        super().go_to_rest(joint_reset)
