import queue
import threading
import time


class VideoCapture:
    def __init__(self, cap,type, name=None):
        if name is None:
            name = cap.name
        self.type=type
        self.name = name
        self.q = queue.Queue()
        self.cap = cap
        self.t = threading.Thread(target=self._reader)
        self.t.daemon = True
        self.enable = True
        if self.type == "oak-d":
            cap.connect()
        else:
            raise NotImplementedError
        self.t.start()

        # read frames as soon as they are available, keeping only most recent one

    def _reader(self):
        while self.enable:
            time.sleep(0.01)
            if self.type == "oak-d":
                frame = self.cap._get_sensors()["rgb"]
                ret = True
            else:
                ret, frame = self.cap.read()
            if not ret:
                break
            if not self.q.empty():
                try:
                    self.q.get_nowait()  # discard previous (unprocessed) frame
                except queue.Empty:
                    pass
            self.q.put(frame)

    def read(self):
        return self.q.get(timeout=5)

    def close(self):
        self.enable = False
        self.t.join()
        self.cap.close()
