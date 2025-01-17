





import numpy as np
import torch
from flask import Flask, request, jsonify
from scipy.spatial.transform import Rotation as R
from absl import app, flags


from franka_env.envs.peg_env.config import PegEnvConfig
from robot_servers.polymetis_interface_noG import RepFrankaGripperServer, RepFrankaServer




def __init__(self,ip, port, gripper_port, gripper_type, reset_joint_target : torch.Tensor,config
                 ): 
        robot = RepFrankaServer(ip, port, reset_joint_target, config)


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


def main(_):
    
    CONFIG = PegEnvConfig

    ip=CONFIG.ROBOT_IP
    port=CONFIG.ROBOT_PORT
    gripper_port = CONFIG.GRIPPER_PORT
    gripper_type=CONFIG.GRIPPER_TYPE
    reset_joint_target = CONFIG.RESET_JOINT_TARGET


    robot = RepFrankaServer(ip, port, CONFIG.RESET_JOINT_TARGET,CONFIG)


    webapp = Flask(__name__)



    if gripper_type == "Robotiq":
        raise NotImplementedError("Gripper Type Not Implemented")
        # from robot_servers.robotiq_gripper_server import RobotiqGripperServer

        gripper_server = RobotiqGripperServer(gripper_ip=GRIPPER_IP)
    elif gripper_type == "Franka":
        gripper = RepFrankaGripperServer(ip, gripper_port)
    elif gripper_type == "None":
        pass
    else:
        raise NotImplementedError("Gripper Type Not Implemented")

    robot.start_impedance()










    @webapp.route("/startimp", methods=["POST"])
    def start_impedance():
        robot.start_impedance()
        return "Started impedance"

    #for Stopping impedance


    @webapp.route("/stopimp", methods=["POST"])
    def stop_impedance():
        robot.stop_impedance() 
        return "Stopped impedance"

    # for Getting Pose


    @webapp.route("/getpos", methods=["POST"])
    def get_pos():
        return jsonify({"pose": robot.get_pos().tolist()})

    @webapp.route("/getpos_euler", methods=["POST"])
    def get_pos_euler():
        pos = get_pos()
        r = R.from_quat(pos[3:])
        euler = r.as_euler("xyz")
        return jsonify({"pose": euler.tolist()})



    @webapp.route("/getvel", methods=["POST"])
    def get_vel():
        return jsonify({"vel": np.array(robot.get_vel()).tolist()})


    @webapp.route("/getforce", methods=["POST"])
    def get_force():
        return jsonify({"force": np.array(robot.get_force()).tolist()})
   


    @webapp.route("/gettorque", methods=["POST"])
    def get_torque():
        return jsonify({"torque": np.array(robot.get_torque()).tolist()})
    

    @webapp.route("/getq", methods=["POST"])
    def get_q():
        return jsonify({"q": np.array(robot.get_q()).tolist()})
    
    

    @webapp.route("/getdq", methods=["POST"])
    def get_dq():
        return jsonify({"dq": np.array(robot.get_dq).tolist()})
   

    @webapp.route("/getjacobian", methods=["POST"])
    def get_jacobian():
        jacobian = robot._set_jacobian()
        return jsonify({"jacobian": np.array(jacobian).tolist()})

    

    @webapp.route("/get_gripper", methods=["POST"])
    def get_gripper():
        # Route for getting gripper distance
        gripper= gripper.get_pos()
        return jsonify({"gripper": gripper})
        





    # Route for Running Joint Reset
    @webapp.route("/jointreset", methods=["POST"])
    def joint_reset():
    # robot_server.clear()
    # robot_server.reset_joint()
        robot.reset_joint()
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
        gripper.open()
        return "Opened"


    # for Closing the Gripper
    @webapp.route("/close_gripper", methods=["POST"])
    def close():
        print("close")
        gripper.close()
        return "Closed"


    # Route for moving the gripper
    @webapp.route("/move_gripper", methods=["POST"])
    def move_gripper():
        pos = request.json
        pos = np.clip(int(pos), 0, 255)  # 0-255
        print(f"move gripper to {pos}")
        gripper.move(pos)
        return "Moved Gripper"



    # # Route for Clearing Errors (Communcation constraints, etc.)
    @webapp.route("/clearerr", methods=["POST"])
    def clear():
        # robot_server.clear()
        return "Clear"
     



    # Route for Sending a pose command [x,y,z, q1,q2,q3,q4]
    @webapp.route("/pose", methods=["POST"])
    def pose():
        pos= torch.Tensor(request.json["arr"])
        robot.move(pos)
        return "Moved"



    # Route for getting all state information
    @webapp.route("/getstate", methods=["POST"])
    def get_state():
        robot._set_currpos()
        return jsonify(
            {
                "pose": robot.pos.tolist(),
                "vel": robot.vel.tolist(),
                "force": robot.force.tolist(),
                "torque": robot.torque.tolist(),
                "q": robot.q.tolist(),
                "dq": robot.dq.tolist(),
                "jacobian": robot.jacobian.tolist(),
                "gripper_pos":gripper.get_pos(),
            }
        )




    # Route for updating compliance parameters
    # TODO robably important for controller and optimised movement
    @webapp.route("/update_param", methods=["POST"])
    def update_param():
        # reconf_client.update_configuration(request.json)
        return "Updated compliance parameters"
        

    webapp.run(host="0.0.0.0")


if __name__ == "__main__":
    app.run(main)