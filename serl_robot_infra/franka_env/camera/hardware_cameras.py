import cv2
from abc import abstractmethod
from typing import Any, Optional
import time
from pathlib import Path
from multiprocessing import Process

# async cam imports
from bisect import bisect_right
from datetime import datetime
from typing import Any, Optional, TypeVar, Generic, List, Type
from multiprocessing import Event
from multiprocessing.managers import BaseManager
from time import sleep
from tempfile import TemporaryDirectory
import shutil

# I installed this through:
#  pip install --trusted-host pypi.python.org moviepy
#  pip install imageio-ffmpeg
from moviepy.editor import VideoFileClip

from franka_env.camera.hardware_devices import DiscreteDevice, ContinuousDevice

class DiscreteCamera(DiscreteDevice):
    """
    This class acts as a generalization for cameras, whose recording is captured frame by frame. It implements the method `cam.store_last_frame(dir, title)`.

    Additionally, this class inherits from `DiscreteDevice`, so its functionality is also included.
    """

    def __init__(self, device_id: str, name: Optional[str] = None, height: int = 512, width: int = 512, start_frame_latency: int = 0) -> None:
        super().__init__(device_id, name if name else f"discrete_cam_{device_id}", start_frame_latency)
        self.format = '.png'
        self.height, self.width = height, width

    @abstractmethod
    def _get_sensors(self) -> dict[str, Any]:
        """
        Prompts the camera to output a single frame. Is overwritten by subclass.
        Output should have the following format: `{'time': timestamp, 'rgb': rgb_vals, 'd' [opt]: depth_vals}`

        Returns:
        -------
        - `sensor_data` (dict): Sensor data in the format `{'time': timestamp, 'rgb': rgb_vals, ...}`.
            
        """
        pass

    def get_format(self) -> str:
        return self.format

    def store_last_frame(self, directory: Path, filename: str):
        """
        Stores the last frame received by camera (only the RGB data) as a `self.format` (default: ".png").

        Parameters:
        ----------
        - `directory` (Path): Directory, where last frame should be stored.
        - `filename` (str): Title of the frame.
        """
        sensor_data = self._get_sensors()
        img = sensor_data["rgb"]
        self.timestamps.append(sensor_data["time"])
        resized_img = cv2.resize(img, (self.width, self.height))
        cvt_color_img = cv2.cvtColor(resized_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(
            str(directory / f"{filename}") + self.format,
            cvt_color_img,
        )
    
    @staticmethod
    @abstractmethod
    def get_devices(amount: int, height: int = 512, width: int = 512, type="discrete", **kwargs) -> list['DiscreteCamera']:
        """
        Finds and returns specific amount of instances of this class. Is overwritten by subclass.

        Parameters:
        ----------
        - `amount` (int): Maximum amount of instances to be found. Leaving out `amount` may return all instances (not always).
        - `height` (int): Pixel-height of captured frames. Default: `512`
        - `width` (int): Pixel-width of captured frames. Default: `512`
        - `**kwargs`: Arbitrary keyword arguments.
        
        Returns:
        --------
        - `devices` (list): List of found devices. If no devices are found, `[]` is returned.
        """
        print(f"Looking for {'up to ' + str(amount) if amount != -1 else 'all'} {type} cameras to capture {height}x{width} frames.")




class ContinuousCamera(ContinuousDevice):

    def __init__(self, device_id: str, name: Optional[str] = None, height: int = 512, width: int = 512, default_fps: float = 20, cut_ending=True, **kwargs) -> None:
        super().__init__(device_id, name if name else f"continuous_cam_{device_id}")
        self.format = '.mp4'
        self.height, self.width = height, width
        self.latency = 0.0 # in s
        self.default_fps = default_fps
        self.cut_ending = cut_ending

        self.frame_extraction_processes = []

    def start_recording(self) -> bool:
        self.recording_start = time.time() + self.latency # timestamp, where recording actually started
        return True

    def stop_recording(self) -> bool:
        self.recording_stop = time.time() # timestamp, where recording SHOULD end
        return True
    
    def store_recording(self, directory, filename = None, timestamps = None):
        filename = filename if filename else f"{self.name}_recording"
        video_file = str(directory / filename) + self.format
        self._store_video(video_file)

        if timestamps:
            duration = timestamps[-1] - timestamps[0]
            print(f"duration: {duration}, timestamps length: {len(timestamps)} => fps: {len(timestamps)/duration}")
            self.__extract_frames_at_timestamps(video_file, directory, timestamps, self.recording_start)
        else:
            self.__extract_frames(video_file, directory, self.default_fps, self.recording_start, self.recording_stop, self.cut_ending)
        return True


    @abstractmethod
    def _store_video(self, video_file: str):
        pass


    def __extract_frames_at_timestamps(self, video_file, directory, timestamps, recording_start):

        def convert_to_images():
            vidcap = cv2.VideoCapture(video_file)
            idx = 0
            for timestamp in timestamps:
                ms_time = max((timestamp - recording_start) * 1000, 0) # relative video position in ms
                vidcap.set(cv2.CAP_PROP_POS_MSEC,ms_time)
                success,image = vidcap.read()
                if success:
                    resized_img = cv2.resize(image, (self.width, self.height))
                    cv2.imwrite(str(directory / f"{idx}.png"), resized_img)       # save frame as png file
                    idx += 1
        
        process = Process(target=convert_to_images)
        process.start()
        self.frame_extraction_processes.append(process)


    def __extract_frames(self, video_file, directory, fps, recording_start, recording_stop, cut_ending=True):
    
        def convert_to_images():
            if cut_ending:
                duration = recording_stop - recording_start # in seconds
                clip = VideoFileClip(video_file).subclip(0, duration)
            else:
                clip = VideoFileClip(video_file)
            clip.write_images_sequence(str(directory / '%d.png'), fps=fps, logger=None) # logger='bar'
        
        process = Process(target=convert_to_images)
        process.start()
        self.frame_extraction_processes.append(process)

    
    def close(self) -> bool:
        for process in self.frame_extraction_processes:
            process.join()
        return True


    @staticmethod
    @abstractmethod
    def get_devices(amount: int, height: int = 512, width: int = 512, type="continuous", **kwargs) -> list['DiscreteCamera']:
        """
        Finds and returns specific amount of instances of this class. Is overwritten by subclass.

        Parameters:
        ----------
        - `amount` (int): Maximum amount of instances to be found. Leaving out `amount` may return all instances (not always).
        - `height` (int): Pixel-height of captured frames. Default: `512`
        - `width` (int): Pixel-width of captured frames. Default: `512`
        - `**kwargs`: Arbitrary keyword arguments.
        
        Returns:
        --------
        - `devices` (list): List of found devices. If no devices are found, `[]` is returned.
        """
        print(f"Looking for {'up to ' + str(amount) if amount != -1 else 'all'} {type} cameras to capture {height}x{width} frames.")


# author of class: TimWindecker
T = TypeVar("T", bound=DiscreteCamera)
class AsynchronousCamera(ContinuousCamera, Generic[T]):
    """
    This class is a wrapper for a DiscreteCamera to act as a ContinuousCamera by running it in a separate process.
    """

    def __init__(self, camera_class: Type[T], capture_interval=0, **kwargs):
        super().__init__(
            **kwargs
        )

        # Create event to signal that the process should stop
        self._stop = Event()

        # Define custom manager to share the camera object between processes
        class CameraManager(BaseManager):
            pass
        V = TypeVar("V")
        class Container(Generic[V]):
            def __init__(self, value: V):
                self._value = value
            def get_value(self) -> V:
                return self._value
        CameraManager.register("Camera", camera_class, exposed=("_connect", "_failed_connect", "close", "store_last_frame", "get_format"))
        CameraManager.register("CaptureInterval", Container[int])
        CameraManager.register("TempDirectoryPath", Container[str])

        # Start manager and create shared objects
        self._manager = CameraManager()
        self._manager.start()
        self._proxy_camera = self._manager.Camera(**kwargs)
        self._proxy_capture_interval = self._manager.CaptureInterval(capture_interval)
        self._temp_dir = TemporaryDirectory(prefix='camera_tmp_')
        self._proxy_temp_dir_path = self._manager.TempDirectoryPath(self._temp_dir.name)
        assert(Path(self._proxy_temp_dir_path.get_value()).exists())

    def _connect(self):
        self._proxy_camera._connect()

    def _failed_connect(self):
        self._proxy_camera._failed_connect()

    def close(self):
        super().close()
        self._proxy_camera.close()

    @staticmethod
    def get_devices(**kwargs) -> list['T']:
        return T.get_devices(**kwargs)

    def _store_video(self, video_file: str):
        raise NotImplementedError

    def start_recording(self) -> bool:
        super().start_recording()

        # Clear temp folder
        self._clear_temp()

        # Start process
        self._process = Process(
            target=self._continuous_capture, 
            args=(
                self._stop,
                self._proxy_camera,
                self._proxy_capture_interval, 
                self._proxy_temp_dir_path
            ), 
            daemon=True
        )
        self._stop.clear()
        self._process.start()

        return True

    def stop_recording(self) -> bool:
        super().stop_recording()
        self._stop.set()
        self._process.join()
        return True

    def store_recording(self, directory: Path, filename: str = None, timestamps: List[float] = None):
        """
        Stores the last frame received by camera (only the RGB data) as a `self.format` (default: ".png").

        Parameters:
        ----------
        - `directory` (Path): Directory, where last frame should be stored.
        - `filename` (str): Title of the frame.
        - `timestamps` (List[float]): Timesamps of the required images as list of seconds since the Unix epoch. 
                                      If the exact timestamp is not available the closest image before is selected.
        """

        if timestamps is None:
            # If no timestamps are given, copy all images
            temp_dir = Path(self._proxy_temp_dir_path.get_value())
            image_format = self._proxy_camera.get_format()
            for image_path in temp_dir.glob(f"*{image_format}"):
                destination_path = directory / f"{image_path.name}"
                shutil.copy(image_path, destination_path)
        else:
            # Extract map from timestamps to image paths
            temp_dir = Path(self._proxy_temp_dir_path.get_value())
            image_format = self._proxy_camera.get_format()
            timestamp_path_map = []
            for image_path in temp_dir.glob(f"*{image_format}"):
                try:
                    timestamp_str = image_path.stem
                    timestamp = datetime.fromisoformat(timestamp_str)
                    timestamp_unix = timestamp.timestamp()
                    timestamp_path_map.append((timestamp_unix, image_path))
                except ValueError:
                    print(f"Skipping file with invalid timestamp: {image_path.name}")

            # Sort map by timestamps
            timestamp_path_map.sort(key=lambda t: t[0])

            # Map desired timestamps to paths
            sorted_timetamps = [t for t, _ in timestamp_path_map]
            desired_indices = [bisect_right(sorted_timetamps, t) - 1 for t in timestamps] # Find index of element <= t
            desired_paths  = [timestamp_path_map[i][1] for i in desired_indices]

            # Copy images
            for i, path in enumerate(desired_paths):
                destination_path = directory / f"{i}{image_format}"
                shutil.copy(path, destination_path)

        # Clear temp folder
        self._clear_temp()

    def delete_recording(self):
        self._clear_temp()

    def _clear_temp(self):
        for file in Path(self._proxy_temp_dir_path.get_value()).rglob("*"):
            file.unlink()

    def _continuous_capture(self, stop_event, camera, capture_interval, dir_path) -> None:
        
        directory = Path(dir_path.get_value())

        while not stop_event.is_set():

            # Capture and store
            filename = datetime.now().isoformat()
            camera.store_last_frame(directory, filename)

            # Sleep    
            sleep(capture_interval.get_value())