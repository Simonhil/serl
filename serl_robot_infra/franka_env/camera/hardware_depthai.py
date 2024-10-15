from threading import Thread
import threading
import numpy as np
# from hardware_base import hardwareBase

import cv2
import logging
import copy
import time
import depthai as dai

from franka_env.camera.hardware_cameras import DiscreteCamera






logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# class DAIThread(Thread):
class DAIThread():
    def __init__(self, device_MxId, **kwargs):
        # Thread.__init__(self, daemon=True)

        pipeline = dai.Pipeline()

        # # Closer-in minimum depth, disparity range is doubled (from 95 to 190):
        # extended_disparity = False
        # # Better accuracy for longer distance, fractional disparity 32-levels:
        # subpixel = False
        # # Better handling for occlusions:
        # lr_check = True
        # monoLeft = pipeline.create(dai.node.MonoCamera)
        # monoRight = pipeline.create(dai.node.MonoCamera)
        # monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        # monoLeft.setCamera("left")
        # monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        # monoRight.setCamera("right")
        # depth = pipeline.create(dai.node.StereoDepth)
        # depth.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        # # Options: MEDIAN_OFF, KERNEL_3x3, KERNEL_5x5, KERNEL_7x7 (default)
        # depth.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
        # depth.setLeftRightCheck(lr_check)
        # depth.setExtendedDisparity(extended_disparity)
        # depth.setSubpixel(subpixel)
        # xoutDepth = pipeline.create(dai.node.XLinkOut)
        # xoutDepth.setStreamName("depth")
        # monoLeft.out.link(depth.left)
        # monoRight.out.link(depth.right)
        # depth.depth.link(xoutDepth.input)

        camRgb = pipeline.create(dai.node.ColorCamera)
        camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
        camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        xoutRgb = pipeline.create(dai.node.XLinkOut)
        xoutRgb.setStreamName("rgb")
        camRgb.video.link(xoutRgb.input)

        self.device_info = dai.DeviceInfo(device_MxId)
        self.pipeline = pipeline

        self.frame = None
        self.timestamp = None
        self.depthFrame = np.zeros((1080, 1920))
        self.should_stop = threading.Event()

        self.device = dai.Device(self.pipeline, self.device_info)
        self.qRgb = self.device.getOutputQueue(name="rgb", maxSize=1, blocking=False)

    def run(self):
        with dai.Device(self.pipeline, self.device_info) as device:
            # Output queue will be used to get the rgb frames from the output defined above
            qRgb = device.getOutputQueue(name="rgb", maxSize=1, blocking=False)
            # qDepth = device.getOutputQueue(name="depth", maxSize=1, blocking=False)
            while not self.should_stop.is_set():
                inRgb = qRgb.get()
                # inDepth = qDepth.get()  # blocking call, will wait until a new data has arrived
                if inRgb is not None:
                    bgr_img = inRgb.getCvFrame()
                    self.frame = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
                    self.timestamp = time.time()
                # if inDepth is not None:
                #     self.depthFrame = inDepth.getFrame()

    def updateImg(self):
        inRgb = self.qRgb.get()
        while inRgb is None:
            inRgb = self.qRgb.get()
            time.sleep(0.1)
        if inRgb is not None:
            bgr_img = inRgb.getCvFrame()
            self.frame = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
            self.timestamp = time.time()


class DepthAI(DiscreteCamera):
    """
    This class can be considered a wrapper class for DepthAI cameras specifically for frame collection.

    This class inherits its functions from `real_robot_env.robot.hardware_devices.DiscreteDevice`.
    """

    def __init__(self, device_id, name = None, height = 512, width = 512, **kwargs):
        super().__init__(device_id, name if name else f"DepthAI_{device_id}", height, width)

        self.timeout = 1 # in seconds
       

    def _connect(self):
        self.thread = DAIThread(self.device_id)
        # self.thread.start()

    def _failed_connect(self):
        self.thread = None

    def _get_sensors(self):
        """
        Prompts the device to output a single frame of the sensor data.
        Output has the following format: `{'time': timestamp, 'rgb': rgb_vals}`

        Depth is currently not supported ('d': NDArray[float64]).

        Returns:
        -------
        - `sensor_data` (dict): Sensor data in the format `{'time': float, 'rgb': Any}`. 
        """

         # get all data from all topics
        self.thread.updateImg()
        last_img = copy.deepcopy(self.thread.frame)
        #last_depth = copy.deepcopy(self.thread.depthFrame) # why was this also self.thread.frame ??
        while last_img is None: #or last_depth is None
            self.thread.updateImg()
            last_img = copy.deepcopy(self.thread.frame)
            #last_depth = copy.deepcopy(self.thread.depthFrame)
            time.sleep(0.1)
            """ {'time':self.thread.timestamp, 'rgb': last_img} """
        return last_img
    
    def apply_commands(self):
        return 0

    def close(self):
        self.thread.should_stop.set()

        # Kevin Kuryshev: I received the Error "AttributeError: 'DAIThread' object has no attribute 'join'" , so I commented this out:
        #self.thread.join()
        return True

    def okay(self):
        # if self.rgb_topic and (self.rgb_sub is None):
        #     print("WARNING: No subscriber found for topic: ", self.rgb_topic)
        #     return False

        # if self.d_topic and (self.d_sub is None):
        #     print("WARNING: No subscriber found for topic: ", self.d_topic)
        #     return False

        # if self.most_recent_pkt_ts is None:
        #     print("WARNING: No packets received yet from the realsense subscibers: ", self.rgb_topic, self.d_topic)
        #     return False
        # else:
        #     now = datetime.datetime.now(datetime.timezone.utc)
        #     okay_age_threshold = datetime.timedelta(seconds=self.timeout)
        #     time_delay = now - self.most_recent_pkt_ts
        #     if time_delay>okay_age_threshold:
        #         print("Significant signal delay: ", time_delay)
        #         return False

        return True

    def reset(self):
        return 0

    @staticmethod
    def get_devices(amount=-1, height: int = 512, width: int = 512, **kwargs) -> list['DepthAI']:
        """
        Finds and returns specific amount of instances of this class.

        Parameters:
        ----------
        - `amount` (int): Maximum amount of instances to be found. Leaving out `amount` or `amount = -1` returns all instances.
        - `height` (int): Pixel-height of captured frames. Default: `512`
        - `width` (int): Pixel-width of captured frames. Default: `512`
        - `**kwargs`: Arbitrary keyword arguments.
        
        Returns:
        --------
        - `devices` (list[DepthAI]): List of found devices. If no devices are found, `[]` is returned.
        """
        super(DepthAI,DepthAI).get_devices(amount, height=height, width=width, type="DepthAI", **kwargs)
        cam_list = dai.Device.getAllAvailableDevices()
        cams = []
        counter = 0
        for device in cam_list:
            if amount != -1 and counter >= amount: break
            cam = DepthAI(device.getMxId(), height=height, width=width)
            cams.append(cam)
            counter += 1
        return cams




class DepthAI_Capture:

    def __init__(self, device_id, name = None, depth=False, **kwargs):
        height = 512
        width = 512
        cam = DepthAI(device_id, name, height, width)
    

    
    def read(self):
        frames = self.pipe.wait_for_frames()
        aligned_frames = self.align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        if self.depth:
            depth_frame = aligned_frames.get_depth_frame()

        if color_frame.is_video_frame():
            image = np.asarray(color_frame.get_data())
            if self.depth and depth_frame.is_depth_frame():
                depth = np.expand_dims(np.asarray(depth_frame.get_data()), axis=2)
                return True, np.concatenate((image, depth), axis=-1)
            else:
                return True, image
        else:
            return False, None





# # Get inputs from user
# def get_args():
#     parser = argparse.ArgumentParser(description="DepthAI Client: Connects to realsense pub.\n"
#     "\nExample: python robot/hardware_realsense.py -r realsense_815412070341/color/image_raw -d realsense_815412070341/depth_uncolored/image_raw")

#     parser.add_argument("-r", "--rgb_topic",
#                         type=str,
#                         help="rgb_topic name of the camera",
#                         default="")
#     parser.add_argument("-d", "--d_topic",
#                         type=str,
#                         help="rgb_topic name of the camera",
#                         default="")
#     parser.add_argument("-v", "--view",
#                         type=None,
#                         help="Choice: CV2",
#                         )
#     return parser.parse_args()

if __name__ == "__main__":
    import cv2

    # args = get_args()
    # print(args)

    print(dai.DeviceInfo())
    MXID = '1844301021D9BF1200'
    # MXID = '1844301071E7AB1200' # second
    rs = DepthAI(name="test cam", device_MxId=MXID)
    rs.connect()

    for i in range(1000000):
        img = rs._get_sensors()
        if img['rgb'] is not None:
            print("Received image{} of size:".format(i), img['rgb'].shape, flush=True)
            cv2.imshow("rgb", img['rgb'])
            cv2.waitKey(1)

        if img['rgb'] is None:
            print(img)

        time.sleep(0.1)

    rs.close()
