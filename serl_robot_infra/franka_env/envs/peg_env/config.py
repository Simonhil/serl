import numpy as np
from torch import tensor
from franka_env.envs.franka_env import DefaultEnvConfig
"""   "top": {
        "serial_number": "14442C10113FE2D200",
        "type": "oak-d" 
       }, """
""" 
   "wrist": {
        "serial_number": "1944301061BB782700",
        "type": "oak-d-sr" 
        } """
   
class PegEnvConfig(DefaultEnvConfig):
    """Set the configuration for FrankaEnv."""

    SERVER_URL: str = "http://127.0.0.1:5000/"
    CAMERAS = {
        "wrist": {
        "serial_number": "1944301061BB782700",
        "type": "oak-d-sr" 
        },
        "front": {
        "serial_number": "184430102111900E00",
        "type": "oak-d-lite" 
       },
      
      
     
    }
    """ ee_pos """
    TARGET_POSE = np.array(
        
      [ 0.7086, -0.0420,  0.2486,  3.1245,  0.0340,  0.2865]
        
    )
    """ + np.array([0.0, 0.0, 0.1, 0.0, 0.0, 0.0]) """
    RESET_POSE = TARGET_POSE + np.array([0.0, 0.0, 0.1, 0.0, 0.0, 0.0])
    REWARD_THRESHOLD: np.ndarray = np.array([0.01,0.01,0.005,6.3,0.2,0.2])
    APPLY_GRIPPER_PENALTY = False
    ACTION_SCALE = np.array([0.02, 0.1, 1])
    RANDOM_RESET = True
    RANDOM_XY_RANGE = 0.005
    RANDOM_RZ_RANGE = np.pi / 12
    ABS_POSE_LIMIT_LOW = np.array(
        [
            TARGET_POSE[0] - RANDOM_XY_RANGE,
            TARGET_POSE[1] - RANDOM_XY_RANGE,
            TARGET_POSE[2],
            TARGET_POSE[3] - 0.01,
            TARGET_POSE[4] - 0.01,
            TARGET_POSE[5] - RANDOM_RZ_RANGE,
        ]
    )
    ABS_POSE_LIMIT_HIGH = np.array(
        [
            TARGET_POSE[0] + RANDOM_XY_RANGE,
            TARGET_POSE[1] + RANDOM_XY_RANGE,
            TARGET_POSE[2] + 0.001,
            TARGET_POSE[3] + 0.001,
            TARGET_POSE[4] + 0.001,
            TARGET_POSE[5] + RANDOM_RZ_RANGE,
        ]
    )
    COMPLIANCE_PARAM = {
        "translational_stiffness": 2000,
        "translational_damping": 89,
        "rotational_stiffness": 150,
        "rotational_damping": 7,
        "translational_Ki": 0,
        "translational_clip_x": 0.003,
        "translational_clip_y": 0.003,
        "translational_clip_z": 0.01,
        "translational_clip_neg_x": 0.003,
        "translational_clip_neg_y": 0.003,
        "translational_clip_neg_z": 0.01,
        "rotational_clip_x": 0.02,
        "rotational_clip_y": 0.02,
        "rotational_clip_z": 0.02,
        "rotational_clip_neg_x": 0.02,
        "rotational_clip_neg_y": 0.02,
        "rotational_clip_neg_z": 0.02,
        "rotational_Ki": 0,
    }
    PRECISION_PARAM = {
        "translational_stiffness": 3000,
        "translational_damping": 89,
        "rotational_stiffness": 300,
        "rotational_damping": 9,
        "translational_Ki": 0.1,
        "translational_clip_x": 0.007,
        "translational_clip_y": 0.007,
        "translational_clip_z": 0.007,
        "translational_clip_neg_x": 0.007,
        "translational_clip_neg_y": 0.007,
        "translational_clip_neg_z": 0.007,
        "rotational_clip_x": 0.05,
        "rotational_clip_y": 0.05,
        "rotational_clip_z": 0.05,
        "rotational_clip_neg_x": 0.05,
        "rotational_clip_neg_y": 0.05,
        "rotational_clip_neg_z": 0.05,
        "rotational_Ki": 0.1,
    }

    ROBOT_IP="141.3.53.63"
    ROBOT_PORT = 50053
    GRIPPER_PORT= 50054
    GRIPPER_TYPE= "Franka"
    RESET_JOINT_TARGET= tensor([ 0.0047,  0.5399, -0.0904, -1.5292,  0.0381,  2.0910, -0.3868])
   

