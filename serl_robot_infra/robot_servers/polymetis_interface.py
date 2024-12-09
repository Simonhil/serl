"""the contence of this file aimes to provide an interface to handle everything otherwise requirering the servwer and redirecting it to polymetis"""













from math import sqrt
from scipy.spatial.transform import Rotation as R
import torch


from robot_servers.config import ConfigParam
from polymetis import GripperInterface, RobotInterface
from robot_servers.helper import pseudo_inverse, saturate_torque_rate

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
    def get_pos(self):
        return self.gripper.get_state().width





class RepFrankaServer:
    """Handles the starting and stopping of the impedance controller
    (as well as backup) joint recovery policy."""


    def __init__(self, robot_ip, port, reset_joint_target : torch.Tensor, 
                 ): 
        
    
        self.translational_clip_min_ = torch.Tensor([-ConfigParam.TRANSLATIONAL_CLIP_NEG_X["default"], -ConfigParam.TRANSLATIONAL_CLIP_NEG_Y["default"], -ConfigParam.TRANSLATIONAL_CLIP_NEG_Z["default"]])
        self.translational_clip_max_ = torch.Tensor([ConfigParam.TRANSLATIONAL_CLIP_X["default"], ConfigParam.TRANSLATIONAL_CLIP_Y["default"], ConfigParam.TRANSLATIONAL_CLIP_Z["default"]])
        self.rotational_clip_min = torch.Tensor([-ConfigParam.ROTATIONAL_CLIP_NEG_X["default"], -ConfigParam.ROTATIONAL_CLIP_NEG_Y["default"], -ConfigParam.ROTATIONAL_CLIP_NEG_Z["default"]])
        self.rotational_clip_max = torch.Tensor([ConfigParam.ROTATIONAL_CLIP_X["default"], ConfigParam.ROTATIONAL_CLIP_Y["default"], ConfigParam.ROTATIONAL_CLIP_Z["default"]])
        
        
        
        
        self.jacobian: torch.Tensor
        self.robot_ip = robot_ip
        self.reset_joint_target = reset_joint_target
        self.position_d_target = torch.zeros(3, dtype=torch.float64)
        self.orientation_d_target = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float64)  # [x, y, z, w]
        self.robot = RobotInterface(
                    ip_address= robot_ip, enforce_version=False, port = port
                )
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

        # Quaternion handling
        
 





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
        
        pos ,ori = self.robot.get_ee_pose()
        self.cart_control = self.robot.start_cartesian_impedance()
        self.robot.update_desired_ee_pose(pos, ori)


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
        self.robot.set_home_pose(self.reset_joint_target)

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
        # self.joint_controller = self.robot.start_joint_impedance()
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
        self.robot.move_to_ee_pose(position, orientation, time_to_go = 3)

    #to do once robot state is certen
    """  def update(self,precission_params): #time, duration, 



        state = self.robot.get_robot_state()
        # coriolis is 7x1 
        coriolisforces : torch.Tensor = self.corioles.forward
        (torch.tensor(state.joint_positions),torch.tensor(state.joint_velocities)     )
        # 6x7 matrix
        jacobian = self._set_jacobian(torch.tensor(state.joint_positions))

        # #reforming: 
        joint_pos = torch.tensor(state.joint_positions)
        joint_vel = torch.tensor(state.joint_velocitie)
        ee_pos, ee_quat = self.robot.get_ee_pose()

        #tau_J_d #7x1 Measured link-side joint torque sensor signals
        tau_J_d = self.robot.joint_torques_computed


        #O_t_EE 'Measured end effector pose in base frame.

        # ee pose??
            # Eigen::Affine3d transform(Eigen::Matrix4d::Map(robot_state.O_T_EE.data()));
            # Eigen::Vector3d position(transform.translation());
            # Eigen::Quaterniond orientation(transform.linear());


        
       
        self.error_[:3] = ee_pos - self.position_d_
        # positon error
       
        self.error_[:3] = torch.minimum(torch.maximum(self.error_[3:],self.translational_clip_min_), self.translational_clip_max_)
        # orientation error
        temp_or =R.from_quat(ee_quat)
        if torch.dot(self.orientation_d_, ee_quat) < 0.0:
            temp_or = - temp_or
        
        temp_or =R.from_quat(ee_quat)
        or_d = R.from_quat(self.orientation_d_)
        error_quat = temp_or.__mul__(or_d)


        # "difference" quaternion
        self.error_[3:] = error_quat.as_quat()[:3]
        # Transform to base frame
        # Clip rotation error
        






        # TODO
        # what is teh equivalent ???????? transformation to base frame??
        # error_.tail(3) << -transform.linear() * error_.tail(3);
        # Clip rotation error
        self.error_[3:] = torch.minimum(torch.maximum(self.error_[3:], self.rotational_clip_min), self.rotational_clip_max)




        # clip error
        self.error_i[:3] =  torch.clamp((self.error_i[:3] + self.error_[:3]), min=-0.1, max=0.1)
        self.error_i[3:] =  torch.clamp((self.error_i[3:] + self.error_[3:]), min=-0.3, max=0.3)



        # // compute control
        # // allocate variables

        tau_task, tau_nullspace, tau_d = torch.zeros(7)

        # // pseudoinverse for nullspace handling
        # // kinematic pseuoinverse

        jacobian_transpose = pseudo_inverse (self.jacobian.T)
    

        tau_task = jacobian.T @ (-self.cartesian_stiffness_ @ self.error_ - self.cartesian_damping_ @ (self.jacobian @ self.dq) - self.Ki_ @ self.error_i)

        # Eigen::Matrix<double, 7, 1> dqe;
        # Eigen::Matrix<double, 7, 1> qe;
        qe, dqe = torch.Tensor(7)
        qe = self.q_d_nullspace_ - self.q
        qe[1] = qe[1] * self.joint1_nullspace_stiffness_
        dqe = self.dq
        dqe[1] =  dqe[1] * 2.0 * sqrt(self.joint1_nullspace_stiffness_)
        ident= torch.eye(7)
        tau_nullspace  (ident - self.jacobian.T @ jacobian_transpose) @ (self. nullspace_stiffness_ * qe - (2.0 * sqrt(self. nullspace_stiffness_)) * dqe)
        
        # // Desired torque
        tau_d =  tau_task + tau_nullspace + coriolisforces
        

        # // Saturate torque rate to avoid discontinuities
        tau_d = saturate_torque_rate(tau_d, tau_J_d)

        # for (size_t i = 0; i < 7; ++i) {
        #     joint_handles_[i].setCommand(tau_d(i));
        # }

        
        #update parameters changed online either through dynamic reconfigure or through the interactive
        # target by filtering
        self.cartesian_stiffness_ = self.filter_params * self.cartesian_stiffness_target_ + (1.0 - self.filter_params_) * self.cartesian_stiffness_
        self.cartesian_damping_ = self.filter_params_ * self.cartesian_damping_target_ + (1.0 - self.filter_params_) * self.cartesian_damping_
        self.nullspace_stiffness_ = self.filter_params_ * self.nullspace_stiffness_target_ + (1.0 - self.filter_params_) * self.nullspace_stiffness_
        self.joint1_nullspace_stiffness_ = self.filter_params_ * self.joint1_nullspace_stiffness_target_ + (1.0 - self.filter_params_) * self.oint1_nullspace_stiffness_
        self.position_d_ = self.filter_params_ * self.position_d_target_ + (1.0 - self.filter_params_) * self.position_d_
        self.orientation_d_ = self.orientation_d_.slerp(self.filter_params_, self.orientation_d_target_)
        self.Ki_ = self.filter_params_ * self.Ki_target_ + (1.0 - self.filter_params_) * self.Ki_
        """













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
        print(ext_force_torque)
        try:
            self.vel = self.jacobian @ self.dq
        except:
            self.vel = np.zeros(6)
            print("Jacobian not set, end-effector velocity temporarily not available")



    def get_pos(self):
       return self.robot.get_ee_pose()

    def get_vel(self):
        return self.vel
     
    def get_force(self):
        return self.force

    def get_torque(self):
        return self.torque

    def get_q(self):
        return self.q

    def get_dq(self):
        return self.dq


###########################################################################################

class RpMainInterface:

    def __init__(self,ip, port, gripper_port, gripper_type, reset_joint_target : torch.Tensor,
                 ): 
        self.robot = RepFrankaServer(ip, port, reset_joint_target)


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
        self.robot.stop_impedance()
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
