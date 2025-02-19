import numpy as np
import time
import requests
import copy
import cv2
import queue
import gym

from franka_env.envs.franka_env import FrankaEnv
from franka_env.utils.rotations import euler_2_quat
from franka_env.envs.box_pushing_env.config import BoxPushingConfig
from gym.spaces import flatten_space, flatten

class FrankaBoxPushingVisual(FrankaEnv):




    def __init__(self, **kwargs):
        super().__init__(**kwargs, config=BoxPushingConfig)

        self.observation_space = gym.spaces.Dict(
            {
                
                "images": gym.spaces.Dict(
                    {
                        
                        # "side": gym.spaces.Box(
                        #     0, 255, shape=(256, 256, 3), dtype=np.uint8
                        # ),
                        "front": gym.spaces.Box(
                            0, 255, shape=(150, 150, 3), dtype=np.uint8
                        ),
                        # "wrist": gym.spaces.Box(
                        #     0, 255, shape=(150, 150, 3), dtype=np.uint8
                        # ),
                    }
                ),
            }
        )
        
        
        self.task_id = 0  # 0 for forward task, 1 for backward task
        """
        the inner safety box is used to prevent the gripper from hitting the two walls of the bins in the center.
        it is particularly useful when there is things you want to avoid running into within the bounding box.
        it uses the intersect_line_bbox function to detect whether the gripper is going to hit the wall
        and clips actions that will lead to collision.
        """
        self.inner_safety_box = gym.spaces.Box(
            self._TARGET_POSE[:3] - np.array([0.07, 0.03, 0.001]),
            self._TARGET_POSE[:3] + np.array([0.07, 0.03, 0.04]),
            dtype=np.float64,
        )

    def intersect_line_bbox(self, p1, p2, bbox_min, bbox_max):
        # Define the parameterized line segment
        # P(t) = p1 + t(p2 - p1)
        tmin = 0
        tmax = 1

        for i in range(3):
            if p1[i] < bbox_min[i] and p2[i] < bbox_min[i]:
                return None
            if p1[i] > bbox_max[i] and p2[i] > bbox_max[i]:
                return None

            # For each axis (x, y, z), compute t values at the intersection points
            if abs(p2[i] - p1[i]) > 1e-10:  # To prevent division by zero
                t1 = (bbox_min[i] - p1[i]) / (p2[i] - p1[i])
                t2 = (bbox_max[i] - p1[i]) / (p2[i] - p1[i])

                # Ensure t1 is smaller than t2
                if t1 > t2:
                    t1, t2 = t2, t1

                tmin = max(tmin, t1)
                tmax = min(tmax, t2)

                if tmin > tmax:
                    return None

        # Compute the intersection point using the t value
        intersection = p1 + tmin * (p2 - p1)

        return intersection

    def clip_safety_box(self, pose):
        pose = super().clip_safety_box(pose)
        # Clip xyz to inner box
        if self.inner_safety_box.contains(pose[:3]):
            # print(f'Command: {pose[:3]}')
            pose[:3] = self.intersect_line_bbox(
                self.currpos[:3],
                pose[:3],
                self.inner_safety_box.low,
                self.inner_safety_box.high,
            )
            # print(f'Clipped: {pose[:3]}')
        return pose

    def crop_image(self, name, image):
        """Crop realsense images to be a square."""
        if name == "side":
            return image[:, 80:560, :]
        elif name == "front":
             return image[100:900, 350:1400, :]
        
        elif name == "top":
            return image[:, 80:560, :]
        
        elif name == "wrist":
            return image[:, :, :]
 
        else:
            return ValueError(f"Camera {name} not recognized in cropping")

    def get_im(self):
        images = {}
        display_images = {}
        for key, cap in self.cap.items():
            try:
                rgb = cap.read()
                cropped_rgb = self.crop_image(key, rgb)
                resized = cv2.resize(
                    cropped_rgb, self.observation_space["images"][key].shape[:2][::-1]
                )
                images[key] = resized[..., ::-1]
                display_images[key] = resized
                if key == "front":
                    display_images[key + "_full"] = cv2.resize(cropped_rgb, (480, 480))
                else:
                    display_images[key + "_full"] = cropped_rgb
            except queue.Empty:
                input(
                    f"{key} camera frozen. Check connect, then press enter to relaunch..."
                )
                self.init_cameras(self.config.CAMERAS)
                return self.get_im()

        self.recording_frames.append(
            np.concatenate([display_images[f"{k}"] for k in self.cap], axis=0)
        )  # only record wrist images since front image is not used for training
        self.img_queue.put(display_images)
        return images

    def task_graph(self, obs=None):
        if obs is None:
            return (self.task_id + 1) % 2

    def set_task_id(self, task_id):
        self.task_id = task_id

    def reset(self, joint_reset=False, **kwargs):
        if self.task_id == 0:
            self.resetpos[1] = self._TARGET_POSE[1] + 0.1
        elif self.task_id == 1:
            self.resetpos[1] = self._TARGET_POSE[1] - 0.1
        else:
            raise ValueError(f"Task id {self.task_id} should be 0 or 1")

        self.curr_path_length = 0
        return dict(images = self.get_im()), {}

    def go_to_rest(self, joint_reset=False):
        """
        Move to the rest position defined in base class.
        Add a small z offset before going to rest to avoid collision with object.
        """
        self._send_gripper_command(1)
        self._update_currpos()
        self._send_pos_command(self.currpos)
        time.sleep(0.5)

        # Move up to clear the slot
        self._update_currpos()
        reset_pose = copy.deepcopy(self.currpos)
        reset_pose[2] += 0.05
        self.interpolate_move(reset_pose, timeout=1)

        # execute the go_to_rest method from the parent class
        super().go_to_rest(joint_reset)
    

    def step(self, action: np.ndarray) -> tuple:
        """standard gym step function."""
        start_time = time.time()
        action = np.clip(action, self.action_space.low, self.action_space.high)
        xyz_delta = action[:3]

        self.nextpos = self.currpos.copy()
        self.nextpos[:3] = self.nextpos[:3] + xyz_delta * self.action_scale[0]

      

        gripper_action = action[6] * self.action_scale[2]

     
        

        self.curr_path_length += 1
        dt = time.time() - start_time
        time.sleep(max(0, (1.0 / self.hz) - dt))

        ob = dict(images = self.get_im())
        reward = 0 #overwrit bu later rewards

        done = self.curr_path_length >= self.max_episode_length or reward == 1
        return ob, reward, done, False, {}
    

class SERLObsWrapperVisual(gym.ObservationWrapper):
    """
    This observation wrapper treat the observation space as a dictionary
    of a flattened state space and the images.
    """

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Dict(
            {
               
                **(self.env.observation_space["images"]),
            }
        )

    def observation(self, obs):
        obs = {
           
            **(obs["images"]),
        }
        return obs
