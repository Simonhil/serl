
import numpy as np
import copy
import time

import requests

def _send_pos_command(self, pos: np.ndarray):
        """Internal function to send position command to the robot."""
        arr = np.array(pos).astype(np.float32)
        data = {"arr": arr.tolist()}
        requests.post(self.url + "pose", json=data)

def move(increment: int) : 
    state = requests.post(url + "getstate").json()
    curpos  =np.array(state ["pose"])
    print(curpos)
    newpos = curpos + [0,0,0.1,0,0,0,0,]
    _send_pos_command(newpos)
    




if __name__ == "__main__":
    url = "http://10.10.10.220/"

#gripper test
    ret =requests.post(url + "getstate").json()
    while (True):
        print(ret)
        requests.post(url + "close_gripper")
        time.sleep(2)
        requests.post(url + "open_gripper")
        time.sleep(2)
        ret = requests.post(url + "getstate").json()
    
#movement test
    while(True):
        move(1)
        time.sleep(3)
        move(-1)


    #currpos = env.currpos
    #newpose = currpos + addition
    #newpose = env.clip_safety_box(newpose)
    #env.interpolate_move(newpose, timeout= 1)
    #env.go_to_rest