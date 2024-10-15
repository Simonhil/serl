# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import time

from polymetis import GripperInterface
import torch

from polymetis import RobotInterface
from torchcontrol.modules.feedforward import Coriolis
from  torchcontrol.transform.rotation import RotationObj



#from polymetis import Gripperinterface

reset_joint_target =  torch.tensor([-0.1568,  0.6025,  0.0381, -1.5723,  0.0052,  2.1417,  0.5684])

if __name__ == "__main__":
    # Initialize robot interface
    robot = RobotInterface(
        ip_address= "141.3.53.63",
        port = 50051,

    )
    gripper = GripperInterface (
        ip_address = "141.3.53.63",
        port = 50052
    ) 
       
    # Reset
    robot.set_home_pose(reset_joint_target)
    robot.go_home()  # Get joint positions
    pos  = robot.get_joint_positions()
    """   ori = RotationObj(ori)
    rot = ori.as_rotvec()
    pos = torch.concat ((pos, rot)) """
    print(f"Current positions: {pos}")
    time.sleep(5)
    # Command robot to pose (move 4th and 6th joint)
    joint_positions_desired = torch.tensor(
        [-0.14, -0.02, 0.05, -1.57, 0.05, 1.50, -0.91]
    )
    robot.go_home()
    print(f"\nMoving joints to: {joint_positions_desired} ...\n")
    state_log = robot.move_to_joint_positions(joint_positions_desired, time_to_go=4.0)

    # Get updated joint positions
    state = robot.get_joint_positions()
    corioles= Coriolis(robot.robot_model)
    test_output= robot.get_ee_pose
    

    print(f"testresult: {test_output}")
  
    
    speed = 0.3
    force = 130
    """   gripper.goto( 0.02, speed, force)

        time.sleep(1)
        gripper.goto(0.7, speed, force)

        time.sleep(3)

    """
    print("complete")
    #robot.go_home()
    #test gripper

    #gripper_state = gripper.get_state()
    #gripper.goto(width=0.01, speed=0.05)
    #gripper.grasp(speed=0.05, force=0.1)
    