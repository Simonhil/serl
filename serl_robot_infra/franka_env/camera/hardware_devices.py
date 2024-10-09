"This file contains the abstract classes necessary for all types of devices for recording."

from abc import ABC, abstractmethod
from typing import Optional

from pathlib import Path


class RecordingDevice(ABC):
    """
    This class acts as an interface for all external devices (cameras and microphones) used for recording.
    All instances of `RecordingDevice` can be connected to (`device.connect()`) and disconnected from (`device.close()`).
    Additionally, each concrete subclass of RecordingDevice contains the function `get_devices`, which allows to easily
    find all/a certain amount of connected devices of that subclass and returns a list of instances.
    """

    def __init__(self, device_id: str, name: Optional[str] = None) -> None:
        """
        Instantiates a device.

        Parameters:
        ----------
        - `device_id` (str): Id used for connecting with the device.
        - `name` (Optional[str]): Name of the device. Default: "device_`device_id`"
        """

        self.device_id = device_id
        self.format: str = None
        self.name = name if name else f"device_{device_id}"


    def connect(self) -> bool:
        """
        Connects to this instance.

        Returns:
        --------
        - `success` (bool): Indicates a successful connection.
        """
        print(f"Connecting to {self.name}: ", end="")
        try:
            self._connect()
        except Exception as e:
            print("Failed with exception: ", e)
            self._failed_connect()
            return False
        print("Success")
        return True

    @abstractmethod
    def _connect(self):
        "This method contains the connection logic and has to be overwritten by subclass."
        pass

    def _failed_connect(self):
        "This method performs the logic, in case of a connection failure. It does not have to be overwritten."
        pass


    @abstractmethod
    def close(self) -> bool:
        """
        Closes the connection to this instance. Is overwritten by subclass.

        Returns:
        --------
        - `success` (bool): Indicates a successful disconnection.
        """
        pass


    @staticmethod
    @abstractmethod
    def get_devices(amount: int, type: str = "recording", **kwargs) -> list['RecordingDevice']:
        """
        Finds and returns specific amount of instances of this class. Is overwritten by subclass.

        Parameters:
        ----------
        - `amount` (int): Maximum amount of instances to be found. Leaving out `amount` may return all instances (not always).
        - `type` (str): Type of recording device. Default: `"recording"`
        - `**kwargs`: Arbitrary keyword arguments.
        
        Returns:
        --------
        - `devices` (list): List of found devices. If no devices are found, `[]` is returned.
        """
        print(f"Looking for {'up to ' + str(amount) if amount != -1 else 'all'} {type} devices.")



# Frame Recording

class DiscreteDevice(RecordingDevice):
    """
    This class is an interface for *discrete* recording devices, which are prompted to output their sensor data
    in single frames (`device.get_sensors()`).

    Additionally, this class inherits from `RecordingDevice`, so its functionality is also included.
    """

    def __init__(self, device_id: str, name: Optional[str] = None) -> None:
        """
        Instantiates a discrete device.

        Parameters:
        ----------
        - `device_id` (str): Id used for connecting with the device.
        - `name` (Optional[str]): Name of the device. Default: "discrete_device_`device_id`"
        """
        super().__init__(device_id, name if name else f"discrete_device_{device_id}")

    @abstractmethod
    def store_last_frame(self, directory: Path, filename: str):
        """
        Stores the last frame received by the device at a given directory under the given title.

        Parameters:
        ----------
        - `directory` (str): Directory, where last frame should be stored.
        - `filename` (str): Title of the frame.
        """
        pass

    @abstractmethod
    def _get_sensors(self):
        "Prompts the device to output a single frame of the sensor data. Is overwritten by subclass."
        pass

    @staticmethod
    @abstractmethod
    def get_devices(amount: int, type: str = "discrete", **kwargs) -> list['DiscreteDevice']:
        super(DiscreteDevice,DiscreteDevice).get_devices(amount, type=type, **kwargs)


# Coninuous Recording

class ContinuousDevice(RecordingDevice):
    """
    This class is an interface for *continuous* recording devices, which record their sensory data continuously and
    return the entire recording at the end (instead of frame by frame).
    They can be prompted to start (`device.start_recording()`), stop (`device.stop_recording()`),
    store (`device.store_recording(file_name)`) and delete (`device.delete_recording()`) the recording.

    Additionally, this class inherits from `RecordingDevice`, so its functionality is also included.
    """

    def __init__(self, device_id: str, name: Optional[str] = None) -> None:
        """
        Instantiates a continuous device.

        Parameters:
        ----------
        - `device_id` (str): Id used for connecting with the device.
        - `name` (Optional[str]): Name of the device. Default: "continuous_device_`device_id`"
        """
        super().__init__(device_id, name if name else f"continuous_device_{device_id}")
    
    @abstractmethod
    def start_recording(self) -> bool:
        "Starts the recording on the device (perhaps starts recording process or sends HTTP request). Is overwritten by subclass."
        pass

    @abstractmethod
    def stop_recording(self) -> bool:
        "Stops the recording on the device. Is overwritten by subclass."
        pass

    @abstractmethod
    def store_recording(self, directory: Path, filename: Optional[str] = None) -> bool:
        "Stores the recording at given directory with (optional) given filename. Is overwritten by subclass."
        pass

    @abstractmethod
    def delete_recording(self) -> bool:
        "Discards the recording. Is overwritten by subclass."
        pass

    @staticmethod
    @abstractmethod
    def get_devices(amount: int, type: str = "continuous", **kwargs) -> list['ContinuousDevice']:
        super(ContinuousDevice,ContinuousDevice).get_devices(amount, type=type, **kwargs)