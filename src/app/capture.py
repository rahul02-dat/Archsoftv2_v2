import cv2
import threading
import logging
from queue import Queue
from typing import Optional, Tuple
import numpy as np
import os

logger = logging.getLogger(__name__)


class CameraCapture:
    def __init__(self, config):
        self.config = config.get('camera', {})
        self.rtsp_url = self.config.get('rtsp_url', '')
        self.fps = self.config.get('fps', 30)
        self.width = self.config.get('width', 1280)
        self.height = self.config.get('height', 720)
        
        self.cap = None
        self.frame_queue = Queue(maxsize=10)
        self.latest_frame = None
        self.latest_annotated = None
        self.lock = threading.Lock()
        self.running = False
        self.thread = None
        
        self._connect()
        
    def _connect(self):
        logger.info(f"Connecting to camera: {self.rtsp_url}")
        
        if self.rtsp_url.startswith('rtsp://'):
            os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp'
            
            self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
            
            if not self.cap.isOpened():
                logger.warning("FFmpeg backend failed, trying default backend...")
                self.cap = cv2.VideoCapture(self.rtsp_url)
        else:
            try:
                camera_index = int(self.rtsp_url)
                self.cap = cv2.VideoCapture(camera_index)
            except ValueError:
                self.cap = cv2.VideoCapture(self.rtsp_url)
        
        if not self.cap.isOpened():
            logger.error("Failed to open camera stream")
            raise ConnectionError("Cannot connect to camera")
        
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        
        logger.info(f"Camera connected: {actual_width}x{actual_height} @ {actual_fps}fps")
        
    def start(self):
        if self.running:
            return
            
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        logger.info("Camera capture started")
        
    def _capture_loop(self):
        consecutive_failures = 0
        max_failures = 30
        
        while self.running:
            ret, frame = self.cap.read()
            
            if not ret:
                consecutive_failures += 1
                logger.warning(f"Failed to read frame ({consecutive_failures}/{max_failures})")
                
                if consecutive_failures >= max_failures:
                    logger.error("Too many consecutive failures, reconnecting...")
                    self._reconnect()
                    consecutive_failures = 0
                continue
            
            consecutive_failures = 0
                
            with self.lock:
                self.latest_frame = frame.copy()
                
            if not self.frame_queue.full():
                self.frame_queue.put(frame)
                
    def _reconnect(self):
        self.cap.release()
        import time
        time.sleep(2)
        self._connect()
        
    def get_frame(self) -> Optional[np.ndarray]:
        try:
            return self.frame_queue.get(timeout=1)
        except:
            return None
            
    def get_latest_frame(self) -> Optional[np.ndarray]:
        with self.lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None
            
    def get_latest_annotated(self) -> Optional[np.ndarray]:
        with self.lock:
            return self.latest_annotated.copy() if self.latest_annotated is not None else None
            
    def set_annotated_frame(self, frame: np.ndarray):
        with self.lock:
            self.latest_annotated = frame.copy()
            
    def release(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        if self.cap:
            self.cap.release()
        logger.info("Camera released")