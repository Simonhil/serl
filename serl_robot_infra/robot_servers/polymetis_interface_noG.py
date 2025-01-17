"""the contence of this file aimes to provide an interface to handle everything otherwise requirering the servwer and redirecting it to polymetis"""













from math import sqrt
from scipy.spatial.transform import Rotation as R
import torch


from franka_env.camera.hardware_franka import ControlType, FrankaArm
from robot_servers.config import ConfigParam
from robot_servers.helper import euler_2_quat, pseudo_inverse, quat_2_euler

import numpy as np
import time


from torchcontrol.modules.feedforward import Coriolis





class RepFrankaGripperServer:
    """_summary_ provides an an interface to acomplish the same methods 
    as the FrankaGripperServer but using polimetis

    Args:
        GripperServer (_type_): _description_
    """
    

    def __init__(self, ip, port):
        super().__init__()
        


    def open(self):
        width = 0.09
        speed = 0.3
        force = 130
        
        

    def close(self):
        
        width = 0.01
        speed = 0.3
        epsilon_inner = 1
        epsilon_outer = 1
        force = 130
        

    def move(self, position: int):
        """Move the gripper to a specific position in range [0, 255]"""
      
        width = float(position / (255 * 10))  # width in [0, 0.1]m
        speed = 0.3
        force = 130
        self.gripper.goto(width, speed, force)
       

    # def _update_gripper(self, msg):
    #     """internal callback to get the latest gripper position."""
    #     self.gripper_pos = np.sum(msg.position)
    def get_pos(self):
        return 0





class RepFrankaServer:
    """Handles the starting and stopping of the impedance controller
    (as well as backup) joint recovery policy."""


    def __init__(self, robot_ip, port, reset_joint_target : torch.Tensor,config 
                 ): 
        
    
        self.translational_clip_min_ = torch.Tensor([-ConfigParam.TRANSLATIONAL_CLIP_NEG_X["default"], -ConfigParam.TRANSLATIONAL_CLIP_NEG_Y["default"], -ConfigParam.TRANSLATIONAL_CLIP_NEG_Z["default"]])
        self.translational_clip_max_ = torch.Tensor([ConfigParam.TRANSLATIONAL_CLIP_X["default"], ConfigParam.TRANSLATIONAL_CLIP_Y["default"], ConfigParam.TRANSLATIONAL_CLIP_Z["default"]])
        self.rotational_clip_min = torch.Tensor([-ConfigParam.ROTATIONAL_CLIP_NEG_X["default"], -ConfigParam.ROTATIONAL_CLIP_NEG_Y["default"], -ConfigParam.ROTATIONAL_CLIP_NEG_Z["default"]])
        self.rotational_clip_max = torch.Tensor([ConfigParam.ROTATIONAL_CLIP_X["default"], ConfigParam.ROTATIONAL_CLIP_Y["default"], ConfigParam.ROTATIONAL_CLIP_Z["default"]])
        
        
        
        
        self.jacobian: torch.Tensor
        self.robot_ip = robot_ip
        self.reset_joint_target = reset_joint_target
        self.position_d_target = torch.zeros(3, dtype=torch.float64)
        self.config = config
        
        
        self.orientation_d_target = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float64)  # [x, y, z, w]
        self.arm = FrankaArm(name='p4', ip_address='141.3.53.63', port=50053, control_type=ControlType.CARTESIAN_IMPEDANCE_CONTROL)
        assert self.arm.connect(home_pose= reset_joint_target), f"connection to robot failed"
        
        
        self.robot = self.arm.robot
        self.robot_model = self.robot.robot_model

        self.corioles= Coriolis(self.robot_model)
        self.error_ =torch.zeros(6)
        self.error_i = torch.zeros(6)
        self.jacobian_array = torch.zeros(42, dtype=torch.float64)
        self.filter_params = 0.005
        self.nullspace_stiffness = 20.0
        self.nullspace_stiffness_target = 20.0
        self.joint1_nullspace_stiffness = 20.0
        self.joint1_nullspace_stiffness_target = 20.0
        self.delta_tau_max = 1.0
        self.cartesian_stiffness = torch.zeros((6, 6), dtype=torch.float64)
        self.cartesian_stiffness_target = torch.zeros((6, 6), dtype=torch.float64)
        self.cartesian_damping = torch.zeros((6, 6), dtype=torch.float64)
        self.cartesian_damping_target = torch.zeros((6, 6), dtype=torch.float64)
        self.Ki = torch.zeros((6, 6), dtype=torch.float64)
        self.Ki_target = torch.zeros((6, 6), dtype=torch.float64)
        # Created from the input parameter
        self.q_d_nullspace = torch.zeros(7, dtype=torch.float64)

        
 

        #clipping params 

        self.translational_clips = []
        self.translational_clips_neg = []
        self.rotational_clips = []
        self.rotational_clips_neg = []


        for key, value in self.config.PRECISION_PARAM.items():
            if "translational_clip_" in key and "neg" not in key:
                self.translational_clips.append(value)
            elif "translational_clip_neg" in key:
                self.translational_clips_neg.append(value)
            elif "rotational_clip_" in key and "neg" not in key:
                self.rotational_clips.append(value)
            elif "rotational_clip_neg" in key:
                self.rotational_clips_neg.append(value)





        # self.eepub = rospy.Publisher(
        #     "/cartesian_impedance_controller/equilibrium_pose",
        #     geom_msg.PoseStamped,
        #     queue_size=10,
        # )
        # self.resetpub = rospy.Publisher(
        #     "/franka_control/error_recovery/goal", ErrorRecoveryActionGoal, queue_size=1
        # )
        # self.jacobian_sub = rospy.Subscriber(
        #     "/cartesian_impedance_controller/franka_jacobian",
        #     ZeroJacobian,
        #     self._set_jacobian,
        # )
        # self.state_sub = rospy.Subscriber(
        #     "franka_state_controller/franka_states", FrankaState, self._set_currpos
        # )






    def start_impedance(self):
        """Launches the impedance controller"""
        self.robot.start_cartesian_impedance()
       

    #replaces the current impedance controller thereby stopping it
    def stop_impedance(self):
        """Stops the impedance controller"""
        self.robot.start_cartesian_impedance()
        time.sleep(1)

    #todo
    # def clear(self):
    #     """Clears any errors"""
    #     msg = ErrorRecoveryActionGoal()
    #     self.resetpub.publish(msg)


    def reset_joint(self):
        """Resets Joints (needed after running for hours)"""
        # First Stop impedance
        try:
            self.arm.robot.move_to_joint_positions(self.reset_joint_target)
            #self.clear()
        except:
            print("impedance Not Running")
        self.start_impedance()
        time.sleep(1)
        print("awake")
      


    def move(self, pose: list):
        """Moves to a pose: [x, y, z, qx, qy, qz, qw]"""
        assert len(pose) == 7

        curpos, curquat = self.robot.get_ee_pose()
        position, orientation = self.clipping(pose, curpos, curquat)
        self.robot.update_desired_ee_pose(position, orientation)





    def clipping(self, action,curpose,curquat):

        pos_d = torch.Tensor(action[:3])

       
        new_pose = np.clip(pos_d, (curpose - torch.Tensor(self.translational_clips_neg)), (curpose + torch.Tensor(self.translational_clips)))

        actioneuler = quat_2_euler(action[3:])

        rot = torch.tensor(quat_2_euler(curquat))
        new_rot = np.clip(actioneuler, (rot- torch.Tensor(self.rotational_clips_neg)), (rot + torch.Tensor(self.rotational_clips)))

        return new_pose, torch.Tensor(euler_2_quat(new_rot))


   

    def _set_jacobian(self, joint_angles):
        self.jacobian = self.robot.robot_model.compute_jacobian(joint_angles)
        return self._set_jacobian
       





    #todo after print status
    def _set_currpos(self):
        #Last commanded end effector pose of motion generation in base frame.
        #Pose is represented as a 4x4 matrix in column-major format. 
        state = self.robot.get_robot_state()
        self.pos, self.orientation = self.robot.get_ee_pose()
        self.pos = torch.concat((self.pos, self.orientation))
        self.dq = torch.Tensor(state.joint_velocities) #joint velocity
        self.q = torch.Tensor(state.joint_positions)# joint angles
        
        #TODO Is that right
        self._set_jacobian(self.q)
        temp_Jac = self.jacobian
        temp_ext = torch.Tensor(state.motor_torques_external)
        ext_force_torque = pseudo_inverse(temp_Jac.T) @ temp_ext
        self.force = ext_force_torque[:3]
        self.torque = ext_force_torque[3:]
   
        try:
            self.vel = self.jacobian @ self.dq
        except:
            self.vel = np.zeros(6)
            print("Jacobian not set, end-effector velocity temporarily not available")



    def get_pos(self):
       return self.robot.get_ee_pose()

    def get_vel(self):
        self._set_currpos()
        return self.vel
     
    def get_force(self):
        self._set_currpos()
        return self.force

    def get_torque(self):
        self._set_currpos()
        return self.torque

    def get_q(self):
        self._set_currpos()
        return self.q

    def get_dq(self):
        self._set_currpos()
        return self.dq


###########################################################################################

class RpMainInterface:

    def __init__(self,ip, port, gripper_port, gripper_type, reset_joint_target : torch.Tensor,config
                 ): 
        self.robot = RepFrankaServer(ip, port, reset_joint_target, config)


        if gripper_type == "Robotiq":
            raise NotImplementedError("Gripper Type Not Implemented")
            # from robot_servers.robotiq_gripper_server import RobotiqGripperServer

            gripper_server = RobotiqGripperServer(gripper_ip=GRIPPER_IP)
        elif gripper_type == "Franka":
            self.gripper = RepFrankaGripperServer(ip, gripper_port)
        elif gripper_type == "None":
            pass
        else:
            raise NotImplementedError("Gripper Type Not Implemented")

        self.robot.start_impedance()
#     reconf_client = ReconfClient(
#         "cartesian_impedance_controllerdynamic_reconfigure_compliance_param_node"
#     )

    #for Starting impedance

    
    def start_impedance(self):
        
        self.robot.start_impedance(self.target_pos, self.target_or)
  
        return "Started impedance"

    #for Stopping impedance

    def stop_impedance(self):
        """  self.robot.stop_impedance() """
        pass
        return "Stopped impedance"

    # for Getting Pose

    def get_pos(self):
        # return jsonify({"pose": np.array(robot_server.pos).tolist()})
        return self.robot.get_pos()


    def get_pos_euler(self):
        pos = self.get_pos()
        r = R.from_quat(pos[3:])
        euler = r.as_euler("xyz")
        return euler

    def get_vel(self):
        # return jsonify({"vel": np.array(robot_server.vel).tolist()})
        return self.robot.get_vel()

   
    def get_force(self):
        # return jsonify({"force": np.array(robot_server.force).tolist()})
        return self.robot.get_force()

    
    def get_torque(self):
        # return jsonify({"torque": np.array(robot_server.torque).tolist()})
        return self.robot.get_torque()

    def get_q(self):
        # return jsonify({"q": np.array(robot_server.q).tolist()})
        return self.robot.get_q()

    def get_dq(self):
        # return jsonify({"dq": np.array(robot_server.dq).tolist()})
        return self.robot.get_dq
    
    def get_jacobian(self):
        # return jsonify({"jacobian": np.array(robot_server.jacobian).tolist()})
        return self.robot._set_jacobian()

    # Route for getting gripper distance
    def get_gripper(self):
        # return jsonify({"gripper": gripper_server.gripper_pos})
        return self.gripper.get_pos()






    # Route for Running Joint Reset
  
    def joint_reset(self):
        # robot_server.clear()
        # robot_server.reset_joint()
        self.robot.reset_joint()
        return "Reset Joint"




    #nly needed with robotiq
    # # Route for Activating the Gripper only needed with robotiq
    # @webapp.route("/activate_gripper", methods=["POST"])
    # def activate_gripper():
    #     print("activate gripper")
    #     gripper_server.activate_gripper()
    #     return "Activated"
    #
    # # Route for Resetting the Gripper. It will reset and activate the gripper
    # @webapp.route("/reset_gripper", methods=["POST"])
    # def reset_gripper():
    #     print("reset gripper")
    #     gripper_server.reset_gripper()
    #     return "Reset"

    #  for Opening the Gripper
    
    def open(self):
        print("open")
        # gripper_server.open()
        self.gripper.open()
        return "Opened"

    # for Closing the Gripper
   
    def close(self):
        print("close")
        self.gripper.close()
        return "Closed"

    # Route for moving the gripper
  
    def move_gripper(self, pos):
       
        pos = np.clip(int(pos), 0, 255)  # 0-255
        print(f"move gripper to {pos}")
        self.gripper.move(pos)
        return "Moved Gripper"

    # # Route for Clearing Errors (Communcation constraints, etc.)
   
    def clear(self):
        # robot_server.clear()
        # return "Clear"
        pass

    # Route for Sending a pose command [x,y,z, q1,q2,q3,q4]

    def pose(self,pos):
        self.robot.move(pos)
        return "Moved"

    # Route for getting all state information
  
    def get_state(self):
        self.robot._set_currpos()
        return (
            {
                "pose": self.robot.pos.tolist(),
                "vel": self.robot.vel.tolist(),
                "force": self.robot.force.tolist(),
                "torque": self.robot.torque.tolist(),
                "q": self.robot.q.tolist(),
                "dq": self.robot.dq.tolist(),
                "jacobian": self.robot.jacobian.tolist(),
                "gripper_pos":self.gripper.get_pos(),
            }
        )
    



    # Route for updating compliance parameters
    # TODO robably important for controller and optimised movement
    def update_param(self, params):
        # reconf_client.update_configuration(request.json)
        # return "Updated compliance parameters"
        pass
