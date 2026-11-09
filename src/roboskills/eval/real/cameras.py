"""RealSense capture only: RGB, metric depth aligned to colour, and the device's intrinsics, one `Stream` per camera
serial (a bench keeps them by name, the rig's `cameras`: {name: {serial, width?, height?, fps?, depth_scale?}}).

No perception happens here (segmentation, fusion and table fitting are `bot`'s, on the host) and nothing here can move
the robot. Each camera streams continuously on its own thread; `grab(after)` returns its first frameset that arrived
after `after` (time.monotonic()), so an observation never shows the scene before the last motion settled.
"""

import threading
import time

import numpy as np


class Stream:
    """One RealSense: colour and depth at `width` x `height` and `fps`, depth aligned to colour, in metres (0: none),
    times `depth_scale` (the rig's, default 1: a camera whose depth reads short of its colour pose)."""

    def __init__(self, serial, width=640, height=480, fps=30, depth_scale=1.0):
        import pyrealsense2 as rs

        self.serial, self.pipe, cfg = str(serial), rs.pipeline(), rs.config()
        cfg.enable_device(self.serial)
        cfg.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
        cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        profile = self.pipe.start(cfg)
        self.depth_scale = float(depth_scale)
        self.scale = profile.get_device().first_depth_sensor().get_depth_scale() * self.depth_scale
        i = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.K = np.array([[i.fx, 0.0, i.ppx], [0.0, i.fy, i.ppy], [0.0, 0.0, 1.0]])
        self.align = rs.align(rs.stream.color)
        self.latest, self.cond, self.running = None, threading.Condition(), True
        self.thread = threading.Thread(target=self._run, daemon=True, name=f"realsense-{self.serial}")
        self.thread.start()

    def _run(self):
        while self.running:
            try:
                frames = self.align.process(self.pipe.wait_for_frames(2000))
            except RuntimeError:  # a dropped frame or a stopped pipeline
                continue
            color, depth = frames.get_color_frame(), frames.get_depth_frame()
            if not color or not depth:
                continue
            rgb = np.asanyarray(color.get_data()).copy()
            z = np.asanyarray(depth.get_data()).astype(np.float32) * self.scale
            with self.cond:
                self.latest = (time.monotonic(), rgb, z)
                self.cond.notify_all()

    def grab(self, after, timeout=2.0):
        """(rgb (H, W, 3) uint8, depth (H, W) float32 m) of the first frameset arriving after `after`."""
        end = time.monotonic() + timeout
        with self.cond:
            while self.latest is None or self.latest[0] <= after:
                if not self.cond.wait(max(0.0, end - time.monotonic())):
                    raise TimeoutError(f"RealSense {self.serial} sent no frame for {timeout} s")
            return self.latest[1], self.latest[2]

    def peek(self):
        """The newest (rgb, depth) without waiting, or None before the first frame (the recorder's)."""
        with self.cond:
            return None if self.latest is None else self.latest[1:]

    def close(self):
        self.running = False
        self.thread.join(timeout=3)
        self.pipe.stop()
