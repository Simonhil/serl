import cv2
from abc import abstractmethod
from typing import Any, Optional

from pathlib import Path

from franka_env.camera.hardware_devices import DiscreteDevice



class DiscreteCamera(DiscreteDevice):
    """
    This class acts as a generalization for cameras, whose recording is captured frame by frame. It implements the method `cam.store_last_frame(dir, title)`.

    Additionally, this class inherits from `DiscreteDevice`, so its functionality is also included.
    """

    def __init__(self, device_id: str, name: Optional[str] = None, height: int = 512, width: int = 512) -> None:
        super().__init__(device_id, name if name else f"discrete_cam_{device_id}")
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

    def store_last_frame(self, directory: Path, filename: str):
        """
        Stores the last frame received by camera (only the RGB data) as a `self.format` (default: ".png").

        Parameters:
        ----------
        - `directory` (Path): Directory, where last frame should be stored.
        - `filename` (str): Title of the frame.
        """
        img = self._get_sensors()["rgb"]
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