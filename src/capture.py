import subprocess
import threading
import time

import cv2


class RealTimeCapture(cv2.VideoCapture):
    """Real Time Streaming Capture.
    这个类必须使用 RealTimeCapture.create 方法创建，请不要直接实例化
    """

    _cur_frame = None
    _reading = False
    schemes = ["rtsp://", "rtmp://"]  # 用于识别实时流

    @staticmethod
    def create(url, schemes=None):
        """实例化&初始化
        rtscap = RealTimeCapture.create("rtsp://example.com/live/1")
        or
        rtscap = RealTimeCapture.create("http://example.com/live/1.m3u8", ["http://"])
        """
        if schemes is None:
            schemes = []
        if not isinstance(url, (str, int)):
            raise ValueError("Invalid URL type")
        rtscap = RealTimeCapture(url)
        rtscap.frame_receiver = threading.Thread(target=rtscap.recv_frame, daemon=True)
        rtscap.schemes.extend(schemes)
        if isinstance(url, str) and url.startswith(tuple(rtscap.schemes)):
            rtscap._reading = True
        elif isinstance(url, int):
            # 这里可能是本机设备
            pass
        else:
            raise ValueError("Invalid URL type")
        return rtscap

    def is_started(self):
        """替代 VideoCapture.isOpened() """
        ok = self.isOpened()
        if ok and self._reading:
            ok = self.frame_receiver.is_alive()
        return ok

    def recv_frame(self):
        """子线程读取最新视频帧方法"""
        while self._reading and self.isOpened():
            ok, frame = self.read()
            if not ok:
                break
            self._cur_frame = frame
        self._reading = False

    def read2(self):
        """读取最新视频帧
        返回结果格式与 VideoCapture.read() 一样
        """
        frame = self._cur_frame
        self._cur_frame = None
        return True, frame

    def start_read(self):
        """启动子线程读取视频帧"""
        if not self.frame_receiver.is_alive():
            self._reading = True
            self.frame_receiver.start()
            self.read_latest_frame = self.read2 if self._reading else self.read

    def stop_read(self):
        """退出子线程方法"""
        self._reading = False
        if self.frame_receiver.is_alive():
            self.frame_receiver.join()