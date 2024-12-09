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

reset_joint_target =  torch.tensor([ 0.0047,  0.5399, -0.0904, -1.5292,  0.0381,  2.0910, -0.3868])

if __name__ == "__main__":
    # Initialize robot interface
    robot = RobotInterface(
        ip_address= "141.3.53.63",
        port = 50053,

    )
    """  gripper = GripperInterface (
        ip_address = "141.3.53.63",
        port = 50052
    ) 
        """
      # Reset
    robot.set_home_pose(reset_joint_target)
    """ robot.go_home()   """# Get joint positions
    pos  = robot.get_joint_positions()
    """ pos, quat = robot.get_ee_pose()
    
    rot = torch.Tensor(quat_2_euler(quat))
    pos = torch.concat ((pos, rot)) """
    print(f"Current positions: {pos}")
    time.sleep(5)
    # Command robot to pose (move 4th and 6th joint)
    joint_positions_desired = torch.tensor([ 0.2311,  0.3784, -0.3684, -1.8208,  0.1226,  2.1922, -0.5033])
    newposQuat = [ 6.3779e-01, -6.7047e-02,  3.4509e-01,  3.0880e+00, -1.4643e-03,
         3.0936e-01]
    new_quat = torch.Tensor(euler_2_quat(torch.Tensor(newposQuat[3:])))
    new_pos = torch.Tensor(newposQuat[:3])

    robot.start_cartesian_impedance()
    """ robot.update_current_policy({
    "ee_pos_desired": new_pos[:3],
    "ee_quat_desired": new_quat,
   
        }) """
    robot.move_to_ee_pose(new_pos, new_quat, )
    robot.move_to_ee_pose(new_pos, new_quat,)

 
   
    print(f"\nMoving joints to: {joint_positions_desired} ...\n")
    """  state_log = robot.move_to_joint_positions(joint_positions_desired)
    """
    # Get updated joint positions
    state = robot.get_joint_positions()
    corioles= Coriolis(robot.robot_model)
    test_output= robot.get_ee_pose()
    

    print(f"testresult: {state}")

    
    speed = 0.3
    force = 130 
    """   gripper.goto( 0.02, speed, force)

        time.sleep(1)
        gripper.goto(0.7, speed, force)

        time.sleep(3)

    """

    print("complete")
    pos  = robot.get_joint_positions()
    pos, quat = robot.get_ee_pose()
    
    rot = torch.Tensor(quat_2_euler(quat))
    pos = torch.concat ((pos, rot))
    print(f"Current positions: {pos}")
    #robot.go_home()
    #test gripper

    #gripper_state = gripper.get_state()
    #gripper.goto(width=0.01, speed=0.05)
    #gripper.grasp(speed=0.05, force=0.1)
    