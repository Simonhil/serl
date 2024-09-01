"""the contence of this file aimes to provide an interface to handle everything otherwise requirering the servwer and redirecting it to polymetis"""













from scipy.spatial.transform import Rotation as R
from polymetis.python.torchcontrol.modules.feedforward import Coriolis
from serl_robot_infra.robot_servers.gripper_server import GripperServer
from polymetis import GripperInterface, RobotInterface
import torch
import numpy as np
import time

class RepFrankaGripperServer(GripperServer):
    """_summary_ provides an an interface to acomplish the same methods 
    as the FrankaGripperServer but using polimetis

    Args:
        GripperServer (_type_): _description_
    """
    

    def __init__(self, id, port):
        super().__init__()
        self.gripper = GripperInterface(
                ip_address=ip,
                port=port
            )
        


    def open(self):
        width = 0.09
        speed = 0.3
        force = 130
        self.gripper.goto(width, speed, force)
        

    def close(self):
        
        width = 0.01
        speed = 0.3
        epsilon_inner = 1
        epsilon_outer = 1
        force = 130
        self.gripper.grasp(speed, force, width, epsilon_inner, epsilon_outer)

    def move(self, position: int):
        """Move the gripper to a specific position in range [0, 255]"""
      
        width = float(position / (255 * 10))  # width in [0, 0.1]m
        speed = 0.3
        force = 130
        self.gripper.goto(width, speed, force)
       

    # def _update_gripper(self, msg):
    #     """internal callback to get the latest gripper position."""
    #     self.gripper_pos = np.sum(msg.position)
    def get_pos():
        pass





class RepFrankaServer:
    """Handles the starting and stopping of the impedance controller
    (as well as backup) joint recovery policy."""


    def __init__(self, robot_ip, gripper_type, port, reset_joint_target : torch.Tensor, position_d_ : torch.Tensor, orientation_d_ : torch.Tensor ): 
        
    

        
        
        
        
        
        
        self.robot_ip = robot_ip
        self.reset_joint_target = reset_joint_target
        self.position_d_ = position_d_
        self.orientation_d_ = orientation_d_

        
        #self.gripper_type = gripper_type

        self.robot = RobotInterface(
                    ip_address= robot_ip, enforce_version=False, port = port
                )
        
        self.robot_model = self.robot.robot_model
        self.corioles= Coriolis(self.robot_module)
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






    def start_impedance(self, target_pos, target_or):
        """Launches the impedance controller"""
        # 
        
        cart_control = self.robot.start_cartesian_impedance()
        self.robot.update_desired_ee_pose(target_pos,target_or)

        time.sleep(5)

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
            self.stop_impedance()
            #self.clear()
        except:
            print("impedance Not Running")
        time.sleep(3)
        #self.clear()

        # Launch joint controller reset
        # set rosparm with rospkg
        # rosparam set /target_joint_positions '[q1, q2, q3, q4, q5, q6, q7]'
        self.robot.et_home_pose(self.reset_joint_target)

        self.joint_controller = self.robot.start_joint_impedance()
            #stdout=subprocess.PIPE,
        time.sleep(1)
        print("RUNNING JOINT RESET")
        #self.clear()

        # Wait until target joint angles are reached
        self.robot.go_home()
        # count = 0
        # time.sleep(1)
        # while not np.allclose(
        #     np.array(self.reset_joint_target) - np.array(self.q),
        #     0,
        #     atol=1e-2,
        #     rtol=1e-2,
        # ):
        #     time.sleep(1)
        #     count += 1
        #     if count > 30:
        #         print("joint reset TIMEOUT")
        #         break

        # Stop joint controller
        print("RESET DONE")
        self.joint_controller = self.robot.start_joint_impedance()
        time.sleep(1)
        #self.clear()
        #print("KILLED JOINT RESET", self.pos)

        # Restart impedece controller
        self.start_impedance()
        print("impedance STARTED")






    def move(self, pose: list):
        """Moves to a pose: [x, y, z, qx, qy, qz, qw]"""
        assert len(pose) == 7
    
        position = torch.Tensor(pose[:3])
        orientation = torch.Tensor(pose[3:])
        self.robot.move_to_ee_position(position, orientation)

    #to do once robot state is certen
    def update(self,precission_params): #time, duration, 

        robot_state = self.robot.get_robot_state()
        coriolisforces : torch.Tensor = self.corioles.forward(     )
        jacobian = self._set_jacobian(          )

        # #reforming: 
        # coriolisforces #7x1
        # jacobian #6x7
        # jointpo #7x1
        # jointvel #7x1
        #tau_J_d #7x1 Measured link-side joint torque sensor signals
        #O_t_EE 'Measured end effector pose in base frame.


        # Eigen::Affine3d transform(Eigen::Matrix4d::Map(robot_state.O_T_EE.data()));
        # Eigen::Vector3d position(transform.translation());
        # Eigen::Quaterniond orientation(transform.linear());


        
        error_ = np.zeros(6)
        error_[:3] = position - position_d_
        error_[:3] = np.clip(error_[:3], translational_clip_min_, translational_clip_max_)


        #weiter bei quarternionen
        



















    #todo after print status
    def _set_currpos(self, msg):
        #Last commanded end effector pose of motion generation in base frame.
        #Pose is represented as a 4x4 matrix in column-major format. 
        tmatrix = np.array(list(msg.O_T_EE)).reshape(4, 4).T 
        r = R.from_matrix(tmatrix[:3, :3])
        pose = np.concatenate([tmatrix[:3, -1], r.as_quat()])
        self.pos = pose
        self.dq = np.array(list(msg.dq)).reshape((7,))#joint velocity
        self.q = np.array(list(msg.q)).reshape((7,))# joint angles
        self.force = np.array(list(msg.K_F_ext_hat_K)[:3])
        self.torque = np.array(list(msg.K_F_ext_hat_K)[3:])
        try:
            self.vel = self.jacobian @ self.dq
        except:
            self.vel = np.zeros(6)
            print("Jacobian not set, end-effector velocity temporarily not available")

    def _set_jacobian(self, joint_angles):
        self.jacobian = self.robot.robot_model.compute_jacobian(joint_angles)
    

    def get_pos():
        pass

    def get_vel():
        pass
     
    def get_force():
        pass

    def get_torque():
        pass
    def get_q():
        pass

    def get_dq():
        pass

    def get_state():
        pass


###########################################################################################

class RpMainInterface:

    def __init__(self, robot_ip, port, gripper_ip, gripper_port, gripper_type, reset_joint_target : torch.Tensor, position_d_ : torch.Tensor,
                 target_pos, target_or, orientation_d_ : torch.Tensor, 
                 ): 
        self.target_pos = target_pos
        self.targget_or = target_or
        self.robot = RepFrankaServer(robot_ip,gripper_type, port, reset_joint_target , position_d_, orientation_d_ )


        if gripper_type == "Robotiq":
            raise NotImplementedError("Gripper Type Not Implemented")
            # from robot_servers.robotiq_gripper_server import RobotiqGripperServer

            gripper_server = RobotiqGripperServer(gripper_ip=GRIPPER_IP)
        elif gripper_type == "Franka":
            self.gripper = RepFrankaGripperServer(gripper_ip, gripper_port)
        elif gripper_type == "None":
            pass
        else:
            raise NotImplementedError("Gripper Type Not Implemented")

        self.robot.start_impedance(target_pos, target_or)
#     reconf_client = ReconfClient(
#         "cartesian_impedance_controllerdynamic_reconfigure_compliance_param_node"
#     )

    #for Starting impedance

    
    def start_impedance(self):
        self.robot.start_impedance(self.target_pos, self.target_or)
        return "Started impedance"

    #for Stopping impedance

    def stop_impedance(self):
        self.robot.stop_impedance()
        return "Stopped impedance"

    # for Getting Pose

    def get_pos(self):
        # return jsonify({"pose": np.array(robot_server.pos).tolist()})
        return self.robot.get_pos()


    def get_pos_euler(self):
        pos = self.get_pos()
        #convert to euler 
        euler = 0
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
        return self.robot.jacobian

    # Route for getting gripper distance
    def get_gripper(self):
        # return jsonify({"gripper": gripper_server.gripper_pos})
        self.gripper.get_pos()





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
   
    def clear():
        # robot_server.clear()
        # return "Clear"
        pass

    # Route for Sending a pose command

    def pose(self,pos):
      
        print("Moving to", pos)
        self.robot.move(pos)
        return "Moved"

    # Route for getting all state information
  
    def get_state(self):
        # return jsonify(
        #     {
        #         "pose": np.array(robot_server.pos).tolist(),
        #         "vel": np.array(robot_server.vel).tolist(),
        #         "force": np.array(robot_server.force).tolist(),
        #         "torque": np.array(robot_server.torque).tolist(),
        #         "q": np.array(robot_server.q).tolist(),
        #         "dq": np.array(robot_server.dq).tolist(),
        #         "jacobian": np.array(robot_server.jacobian).tolist(),
        #         "gripper_pos": gripper_server.gripper_pos,
        #     }
        # )
        return self.robot.get_state()
    



    # Route for updating compliance parameters
   
    def update_param(self, precission_params):
        # reconf_client.update_configuration(request.json)
        # return "Updated compliance parameters"
        self.robot.update(precission_params)



if __name__ == "__main__":
    app.run(main)