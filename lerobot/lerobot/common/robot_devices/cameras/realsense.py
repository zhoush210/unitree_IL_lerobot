"""
This file contains utilities for recording frames from cameras. For more info look at `OpenCVCamera` docstring.
"""

import argparse
import concurrent.futures
import math
import platform
import shutil
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from threading import Thread

import cv2
import numpy as np
from PIL import Image

from lerobot.common.robot_devices.utils import RobotDeviceAlreadyConnectedError, RobotDeviceNotConnectedError
from lerobot.common.utils.utils import capture_timestamp_utc
from lerobot.lerobot.scripts.control_robot_unitree import busy_wait
import pyrealsense2 as rs

# Use 1 thread to avoid blocking the main thread. Especially useful during data collection
# when other threads are used to save the images.
cv2.setNumThreads(1)

# The maximum opencv device index depends on your operating system. For instance,
# if you have 3 cameras, they should be associated to index 0, 1, and 2. This is the case
# on MacOS. However, on Ubuntu, the indices are different like 6, 16, 23.
# When you change the USB port or reboot the computer, the operating system might
# treat the same cameras as new devices. Thus we select a higher bound to search indices.
MAX_OPENCV_INDEX = 60

@staticmethod
def get_connected_devices_serial():
    serials = list()
    for d in rs.context().devices:
        if d.get_info(rs.camera_info.name).lower() != 'platform camera':
            serial = d.get_info(rs.camera_info.serial_number)
            product_line = d.get_info(rs.camera_info.product_line)
            if product_line == 'D400':
                # only works with D400 series
                serials.append(serial)
    serials = sorted(serials)
    return serials


def save_image(img_array, camera_index, frame_index, images_dir):
    img = Image.fromarray(img_array)
    path = str(images_dir) + "/" + "camera_" + str(camera_index).zfill(2) + "_frame_" + str(frame_index).zfill(6) + ".png"
    # path = images_dir / f"camera_{camera_index:02d}_frame_{frame_index:06d}.png"
    # path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, quality=100)



def save_images_from_cameras(
    images_dir: Path, camera_ids: list[int] | None = None, fps=None, width=None, height=None, record_time_s=2
):
    if camera_ids is None:
        camera_ids = get_connected_devices_serial()
    print("camera_ids", camera_ids)
    print("Connecting cameras")
    cameras = []
    for cam_idx in camera_ids:
        camera = OpenCVCamera(cam_idx, fps=fps, width=width, height=height)
        camera.connect()
        print(
            f"OpenCVCamera({camera.camera_index}, fps={camera.fps}, width={camera.width}, "
            f"height={camera.height}, color_mode={camera.color_mode})"
        )
        cameras.append(camera)

    images_dir = Path(images_dir)
    if images_dir.exists():
        shutil.rmtree(
            images_dir,
        )
    images_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving images to {images_dir}")
    frame_index = 0
    start_time = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        while True:
            now = time.perf_counter()

            for camera in cameras:
                # If we use async_read when fps is None, the loop will go full speed, and we will endup
                # saving the same images from the cameras multiple times until the RAM/disk is full.
                image = camera.read() if fps is None else camera.async_read()
                executor.submit(
                    save_image,
                    image,
                    camera.camera_index,
                    frame_index,
                    images_dir,
                )
            if fps is not None:
                dt_s = time.perf_counter() - now
                busy_wait(1 / fps - dt_s)

            if time.perf_counter() - start_time > record_time_s:
                break

            print(f"Frame: {frame_index:04d}\tLatency (ms): {(time.perf_counter() - now) * 1000:.2f}")

            frame_index += 1

    print(f"Images have been saved to {images_dir}")


@dataclass
class OpenCVCameraConfig:
    fps: int | None = None
    width: int | None = None
    height: int | None = None
    color_mode: str = "rgb"

    def __post_init__(self):
        if self.color_mode not in ["rgb", "bgr"]:
            raise ValueError(
                f"Expected color_mode values are 'rgb' or 'bgr', but {self.color_mode} is provided."
            )


class OpenCVCamera:

    def __init__(self, camera_index: int, config: OpenCVCameraConfig | None = None, **kwargs):
        if config is None:
            config = OpenCVCameraConfig()
        # Overwrite config arguments using kwargs
        config = replace(config, **kwargs)

        self.camera_index = camera_index
        self.fps = config.fps
        self.width = config.width
        self.height = config.height
        self.color_mode = config.color_mode
        align_to = rs.stream.color
        self.align = rs.align(align_to)


        self.enable_depth = False

        self.camera = None
        self.is_connected = False
        self.thread = None
        self.stop_event = None
        self.color_image = None
        self.logs = {}

    def init_realsense(self, image=[640, 480], fps=30):

        self.image_shape = image
        self.pipeline = rs.pipeline()
        config = rs.config()
        if self.camera_index is not None:
            config.enable_device(self.camera_index)  
            config.enable_stream(rs.stream.color, image[0], image[1], rs.format.bgr8, fps)  
        if self.enable_depth:
            config.enable_stream(rs.stream.depth, image[0], image[1], rs.format.z16, fps)  

        profile = self.pipeline.start(config)
        self._device = profile.get_device()
        if self._device is None:
            print('pipe_profile.get_device() is None .')
        if self.enable_depth:
            assert self._device is not None
            depth_sensor = self._device.first_depth_sensor()
            self.g_depth_scale = depth_sensor.get_depth_scale()

    def connect(self):
        if self.is_connected:
            raise RobotDeviceAlreadyConnectedError(f"Camera {self.camera_index} is already connected.")
        self.init_realsense(fps=self.fps)
        self.is_connected = True

    def read(self, temporary_color_mode: str | None = None) -> np.ndarray:
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                f"OpenCVCamera({self.camera_index}) is not connected. Try running `camera.connect()` first."
            )

        start_time = time.perf_counter()

        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        if self.enable_depth:
            depth_frame = aligned_frames.get_depth_frame()


        if not color_frame:
            return None, None
        color_image = np.asanyarray(color_frame.get_data())
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        depth_image = np.asanyarray(depth_frame.get_data()) if self.enable_depth else None

        # log the number of seconds it took to read the image
        self.logs["delta_timestamp_s"] = time.perf_counter() - start_time

        # log the utc time at which the image was received
        self.logs["timestamp_utc"] = capture_timestamp_utc()

        return color_image

    def read_loop(self):
        while self.stop_event is None or not self.stop_event.is_set():
            self.color_image = self.read()

    def async_read(self):
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                f"OpenCVCamera({self.camera_index}) is not connected. Try running `camera.connect()` first."
            )

        if self.thread is None:
            self.stop_event = threading.Event()
            self.thread = Thread(target=self.read_loop, args=())
            self.thread.daemon = True
            self.thread.start()

        num_tries = 0
        while self.color_image is None:
            num_tries += 1
            time.sleep(1 / self.fps)
            if num_tries > self.fps and (self.thread.ident is None or not self.thread.is_alive()):
                raise Exception(
                    "The thread responsible for `self.async_read()` took too much time to start. There might be an issue. Verify that `self.thread.start()` has been called."
                )

        return self.color_image

    def disconnect(self):
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                f"OpenCVCamera({self.camera_index}) is not connected. Try running `camera.connect()` first."
            )

        if self.thread is not None and self.thread.is_alive():
            # wait for the thread to finish
            self.stop_event.set()
            self.thread.join()
            self.thread = None
            self.stop_event = None

        # self.camera.release()
        self.camera = None

        self.is_connected = False

    def __del__(self):
        if getattr(self, "is_connected", False):
            self.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Save a few frames using `OpenCVCamera` for all cameras connected to the computer, or a selected subset."
    )
    parser.add_argument(
        "--camera-ids",
        type=int,
        nargs="*",
        default=None,
        help="List of camera indices used to instantiate the `OpenCVCamera`. If not provided, find and use all available camera indices.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Set the number of frames recorded per seconds for all cameras. If not provided, use the default fps of each camera.",
    )
    parser.add_argument(
        "--width",
        type=str,
        default=640,
        help="Set the width for all cameras. If not provided, use the default width of each camera.",
    )
    parser.add_argument(
        "--height",
        type=str,
        default=480,
        help="Set the height for all cameras. If not provided, use the default height of each camera.",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default="outputs/images_from_opencv_cameras",
        help="Set directory to save a few frames for each camera.",
    )
    parser.add_argument(
        "--record-time-s",
        type=float,
        default=2.0,
        help="Set the number of seconds used to record the frames. By default, 2 seconds.",
    )
    args = parser.parse_args()
    save_images_from_cameras(**vars(args))
