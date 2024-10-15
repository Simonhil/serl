import time
from polymetis.gripper_interface import GripperInterface
import torch

from polymetis import RobotInterface
from torchcontrol.modules.feedforward import Coriolis

from robot_servers.polymetis_interface import RpMainInterface

from scipy.spatial.transform import Rotation







#from polymetis import Gripperinterface






ip="141.3.53.63"
rb_port =  50051
g_port = 50052
reset_joint_target =  torch.tensor([-0.1400, -0.0200,  0.0500, -1.5700,  0.0500,  1.5000, -0.9100])
position_d_ = torch.tensor([ 0.6886, -0.0874,  0.2874])
orientation_d_ = torch.tensor([ 0.9416, -0.3357, -0.0228, -0.0138])
target_pos = torch.tensor([ 0.6886, -0.0874,  0.2874])
target_or = torch.tensor([ 0.9416, -0.3357, -0.0228, -0.0138])

if __name__ == "__main__":
    # Initialize robot interface
    robot = RpMainInterface(ip,rb_port,g_port,"Franka", reset_joint_target,)
    
    # Reset
    #robot.go_home()

    # Get joint positions
    print("innitialised")
    #robot.joint_reset()
    positions = robot.get_gripper()
    robot.joint_reset()
    print(f"Current ositions: {positions}")
    currpos = [ 0.670111 ,  -0.02576363 , 0.27626044 , 0.80817068 ,-0.1846139 , -0.53024435,-0.17781684]
    nextpos =  Rotation.from_quat(currpos[3:])
    # Command robot to pose (move 4th and 6th joint)
    """ joint_positions_desired = torch.Tensor(
        [-0.14, -0.02, -0.05, -1.57, 0.05, 1.50, -0.91]
    )
    #robot.joint_reset()

    print(f"\nMoving to: {joint_positions_desired} ...\n")
    delta = torch.tensor([0,0,0.3])
    pos, quat = robot.get_pos()

    robot.pose(torch.cat((position_d_, orientation_d_)))
    

    robot.close()
    time.sleep(1)
    robot.open()
    

    # Get updated joint positions
    test_output= robot.get_state()
    

    print(f"testresult: {test_output}")
  
     """
    #robot.joint_reset()
    robot.stop_impedance()