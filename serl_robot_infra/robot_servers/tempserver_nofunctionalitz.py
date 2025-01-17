





import numpy as np
import torch
from flask import Flask, request, jsonify
from scipy.spatial.transform import Rotation as R
from absl import app, flags


from franka_env.envs.peg_env.config import PegEnvConfig
from robot_servers.polymetis_interface_noG import RepFrankaGripperServer, RepFrankaServer




def __init__(self,ip, port, gripper_port, gripper_type, reset_joint_target : torch.Tensor,config
                 ): 
        pass


def main(_):
    
    CONFIG = PegEnvConfig

    ip=CONFIG.ROBOT_IP
    port=CONFIG.ROBOT_PORT
    gripper_port = CONFIG.GRIPPER_PORT
    gripper_type=CONFIG.GRIPPER_TYPE
    reset_joint_target = CONFIG.RESET_JOINT_TARGET


    webapp = Flask(__name__)



    




    @webapp.route("/startimp", methods=["POST"])
    def start_impedance():
        pass
    #for Stopping impedance


    @webapp.route("/stopimp", methods=["POST"])
    def stop_impedance():
        pass
    # for Getting Pose


    @webapp.route("/getpos", methods=["POST"])
    def get_pos():

        empty_array = np.empty(7)
        return jsonify({"pose": empty_array.tolist()})

    @webapp.route("/getpos_euler", methods=["POST"])
    def get_pos_euler():
       
        euler = np.empty(3)
        return jsonify({"pose": euler.tolist()})



    @webapp.route("/getvel", methods=["POST"])
    def get_vel():
        vel = np.empty(3)
        return jsonify({"vel": vel.tolist()})


    @webapp.route("/getforce", methods=["POST"])
    def get_force():
        euler = np.empty(3)
        return jsonify({"force": euler.tolist()})
   


    @webapp.route("/gettorque", methods=["POST"])
    def get_torque():
        euler = np.empty(3)
        return jsonify({"torque": euler.tolist()})
    

    @webapp.route("/getq", methods=["POST"])
    def get_q():
        euler = np.empty(7)
        return jsonify({"q": euler.tolist()})
    
    

    @webapp.route("/getdq", methods=["POST"])
    def get_dq():
        euler = np.empty(7)
        return jsonify({"dq": euler.tolist()})
   

    @webapp.route("/getjacobian", methods=["POST"])
    def get_jacobian():
        jacobian = [None]
        return jsonify({"jacobian": np.array(jacobian).tolist()})

    

    @webapp.route("/get_gripper", methods=["POST"])
    def get_gripper():
        # Route for getting gripper distance
        gripper= 0
        return jsonify({"gripper": gripper})
        





    # Route for Running Joint Reset
    @webapp.route("/jointreset", methods=["POST"])
    def joint_reset():


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
    @webapp.route("/open_gripper", methods=["POST"])
    def open():
        print("open")
        # gripper_server.open()
   
        return "Opened"


    # for Closing the Gripper
    @webapp.route("/close_gripper", methods=["POST"])
    def close():
        print("close")
 
        return "Closed"


    # Route for moving the gripper
    @webapp.route("/move_gripper", methods=["POST"])
    def move_gripper():
        return "Moved Gripper"



    # # Route for Clearing Errors (Communcation constraints, etc.)
    @webapp.route("/clearerr", methods=["POST"])
    def clear():
        # robot_server.clear()
        return "Clear"
     



    # Route for Sending a pose command [x,y,z, q1,q2,q3,q4]
    @webapp.route("/pose", methods=["POST"])
    def pose():
        return "Moved"



    # Route for getting all state information
    @webapp.route("/getstate", methods=["POST"])
    def get_state():
        # robot._set_currpos()
        # return jsonify(
        #     {
        #         "pose": robot.pos.tolist(),
        #         "vel": robot.vel.tolist(),
        #         "force": robot.force.tolist(),
        #         "torque": robot.torque.tolist(),
        #         "q": robot.q.tolist(),
        #         "dq": robot.dq.tolist(),
        #         "jacobian": robot.jacobian.tolist(),
        #         "gripper_pos":gripper.get_pos(),
        #     }
        # )
        pass




    # Route for updating compliance parameters
    # TODO robably important for controller and optimised movement
    @webapp.route("/update_param", methods=["POST"])
    def update_param():
        # reconf_client.update_configuration(request.json)
        return "Updated compliance parameters"
        

    webapp.run(host="0.0.0.0")


if __name__ == "__main__":
    app.run(main)