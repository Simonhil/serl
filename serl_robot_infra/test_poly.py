# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import time


from polymetis import GripperInterface
import torch

from polymetis import RobotInterface
from robot_servers.helper import euler_2_quat, quat_2_euler
from torchcontrol.modules.feedforward import Coriolis
from  torchcontrol.transform.rotation import RotationObj
from scipy.spatial.transform import Rotation as R



#from polymetis import Gripperinterface

reset_joint_target =  torch.tensor([4.5901e-04,  6.7075e-01, -9.7745e-02, -1.4831e+00,  8.5626e-02,
         2.1018e+00, -5.1625e-01])

if __name__ == "__main__":
    # Initialize robot interface
    robot = RobotInterface(
        ip_address= "141.3.53.63",
        port = 50053,

    )
    gripper = GripperInterface (
        ip_address = "141.3.53.63",
        port = 50054
    ) 
       
      # Reset
    """ robot.move_to_joint_positions(reset_joint_target) """
    """  robot.set_home_pose(reset_joint_target)
    robot.go_home()   # Get joint positions
    pos  = robot.get_joint_positions()
    print(pos)
    """


    print("gripper")


    speed = 0.3
    force = 1 
    state = gripper.get_state()
    print(state)

    gripper.goto(0.08, speed, force)

    time.sleep(1)

    state = gripper.get_state()
    print(state)


    gripper.goto(0.0, speed, force)






    pass

    """ pos, quat = robot.get_ee_pose()
    rot = torch.Tensor(quat_2_euler(quat))
    pos = torch.concat ((pos, rot)) """
    """ print(f"Current positions: {pos}") """
    time.sleep(5)
    # Command robot to pose (move 4th and 6th joint)
    joint_positions_desired = torch.tensor([ 0.2311,  0.3784, -0.3684, -1.8208,  0.1226,  2.1922, -0.5033])



    TARGET_POSE =torch.Tensor(
        
      [ 0.7086, -0.0420,  0.2486,  3.1245,  0.0340,  0.2865]
        
    )

    newposQuat = TARGET_POSE + torch.Tensor([0.0, 0.0, 0.1, 0.0, 0.0, 0.0])
    new_quat = torch.Tensor(euler_2_quat(torch.Tensor(newposQuat[3:])))
    new_pos = torch.Tensor(newposQuat[:3])

    robot.start_cartesian_impedance()
    """ robot.update_current_policy({
    "ee_pos_desired": new_pos[:3],
    "ee_quat_desired": new_quat,
   
        }) """
    """  robot.move_to_ee_pose(new_pos, new_quat, ) """
   
    """ print(f"\nMoving joints to: {joint_positions_desired} ...\n") """
    """  state_log = robot.move_to_joint_positions(joint_positions_desired)
    """
    """  # Get updated joint positions
    state = robot.get_joint_positions()
    corioles= Coriolis(robot.robot_model)
    test_output= robot.get_ee_pose()
    

    print(f"testresult: {state}")

     """

    """   gripper.goto( 0.02, speed, force)

        time.sleep(1)
        gripper.goto(0.7, speed, force)

        time.sleep(3)

    """

    """ pos  = robot.get_joint_positions()
    pos, quat = robot.get_ee_pose()
    
    rot = torch.Tensor(quat_2_euler(quat))
    pos = torch.concat ((pos, rot)) """
    
    #robot.go_home()
    #test gripper

    #gripper_state = gripper.get_state()
    #gripper.goto(width=0.01, speed=0.05)
    #gripper.grasp(speed=0.05, force=0.1)
    