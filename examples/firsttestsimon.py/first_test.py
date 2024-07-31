import gym
import numpy as np
import copy
import time
import franka_first_tests_env

if __name__ == "__main__":
    env = gym.make("FrankaTestSimon -v0")
    env = franka_first_tests_env.FrankaTest(env)

#gripper test
    env._send_gripper_command(1)
    time.sleep(1)
    env._send_gripper_command(0)
    
    
#movement test
    #env.go_to_rest
    #addition = np.array([0, 0.1, 0, 0, 0 ,0])

    #currpos = env.currpos
    #newpose = currpos + addition
    #newpose = env.clip_safety_box(newpose)
    #env.interpolate_move(newpose, timeout= 1)
    #env.go_to_rest