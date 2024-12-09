import time
from franka_env.envs.peg_env.config import PegEnvConfig
from polymetis.gripper_interface import GripperInterface
import torch

from polymetis import RobotInterface
from robot_servers.polymetis_interface_noG import RpMainInterface
from test_poly import euler_2_quat, quat_2_euler
from torchcontrol.modules.feedforward import Coriolis

from scipy.spatial.transform import Rotation







#from polymetis import Gripperinterface






ip="141.3.53.63"
rb_port =  50053
g_port = 50052
reset_joint_target =  torch.tensor([-0.1400, -0.0200,  0.0500, -1.5700,  0.0500,  1.5000, -0.9100])
position_d_ = torch.tensor([ 0.6886, -0.0874,  0.2874])
orientation_d_ = torch.tensor([ 0.9416, -0.3357, -0.0228, -0.0138])
target_pos = torch.tensor([ 0.6886, -0.0874,  0.2874])
target_or = torch.tensor([ 0.9416, -0.3357, -0.0228, -0.0138])

if __name__ == "__main__":
    # Initialize robot interface
    robot = RpMainInterface(ip,rb_port,g_port,"Franka", reset_joint_target, PegEnvConfig)
    
    # Reset


    # Get joint positions
    print("innitialised")

    pos  = robot.get_q()
    pos, quat = robot.get_pos()
    
    rot = torch.Tensor(quat_2_euler(quat))
    pos = torch.concat ((pos, rot))
    print(f"Current positions: {pos}")
    time.sleep(5)
    # Command robot to pose (move 4th and 6th joint)
    joint_positions_desired = torch.tensor([ 0.2311,  0.3784, -0.3684, -1.8208,  0.1226,  2.1922, -0.5033])
    newposQuat = torch.tensor([ 0.7114, -0.0473,  0.2606,  3.0872,  0.0156,  0.0439])
    new_quat = torch.Tensor(euler_2_quat(torch.Tensor(newposQuat[3:])))
    new_pos = torch.Tensor(newposQuat[:3])

    robot.pose(torch.concat((new_pos, new_quat))
        )
    print("done")