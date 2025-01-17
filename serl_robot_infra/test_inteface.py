import time
from franka_env.envs.peg_env.config import PegEnvConfig






from scipy.spatial.transform import Rotation
import requests






#from polymetis import Gripperinterface






ip="141.3.53.63"
rb_port =  50053
g_port = 50052
reset_joint_target =  [-0.1400, -0.0200,  0.0500, -1.5700,  0.0500,  1.5000, -0.9100]
position_d_ = [ 0.6886, -0.0874,  0.2874]
orientation_d_ = [ 0.9416, -0.3357, -0.0228, -0.0138]
target_pos = [ 0.6886, -0.0874,  0.2874]
target_or = [ 0.9416, -0.3357, -0.0228, -0.0138]


url = "http://127.0.0.1:5000/"




if __name__ == "__main__":
    # Initialize robot interface


    # Get joint positions
    print("innitialised")

    pos  = requests.post( url + "/getstate").json()
    requests.post(url + "jointreset")
    print(pos)

    """ pos, quat = robot.get_pos()
    
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
        ) """
    print("done")