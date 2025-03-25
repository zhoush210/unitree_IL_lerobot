"""
Script to convert Aloha hdf5 data to the LeRobot dataset v2.0 format.

Example usage: uv run examples/aloha_real/convert_aloha_data_to_lerobot.py --raw-dir /path/to/raw/data --repo-id <org>/<dataset-name>
"""

import dataclasses
from pathlib import Path
import shutil
from typing import Literal

import h5py
from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.push_dataset_to_hub._download_raw import download_raw
import numpy as np
import torch
import tqdm
import tyro
import os
import json
import cv2
import glob
import re
@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: str | None = None


DEFAULT_DATASET_CONFIG = DatasetConfig()


z1_motors_with_gripper = [
        "kLeftWaist",
        "kLeftShoulder",
        "kLeftElbow",
        "kLeftForearmRoll",
        "kLeftWristAngle",
        "kLeftWristRotate",
        "kLeftGripper",
        "kRightWaist",
        "kRightShoulder",
        "kRightElbow",
        "kRightForearmRoll",
        "kRightWristAngle",
        "kRightWristRotate",
        "kRightGripper",
        ]


g1_motors_with_gripper = [        
        "kLeftShoulderPitch",
        "kLeftShoulderRoll",
        "kLeftShoulderYaw",
        "kLeftElbow",
        "kLeftWristRoll",
        "kLeftWristPitch",
        "kLeftWristyaw",
        "kRightShoulderPitch",
        "kRightShoulderRoll",
        "kRightShoulderYaw",
        "kRightElbow",
        "kRightWristRoll",
        "kRightWristPitch",
        "kRightWristYaw",
        "kLeftGripper",
        "kRightGripper"
        ]

g1_motors_with_dex3 = [        
        "kLeftShoulderPitch",
        "kLeftShoulderRoll",
        "kLeftShoulderYaw",
        "kLeftElbow",
        "kLeftWristRoll",
        "kLeftWristPitch",
        "kLeftWristYaw",
        "kRightShoulderPitch",
        "kRightShoulderRoll",
        "kRightShoulderYaw",
        "kRightElbow",
        "kRightWristRoll",
        "kRightWristPitch",
        "kRightWristYaw",
        "kLeftHandThumb0",
        "kLeftHandThumb1",
        "kLeftHandThumb2",
        "kLeftHandMiddle0",
        "kLeftHandMiddle1",
        "kLeftHandIndex0",
        "kLeftHandIndex1",
        "kRightHandThumb0",
        "kRightHandThumb1",
        "kRightHandThumb2",
        "kRightHandIndex0",
        "kRightHandIndex1",
        "kRightHandMiddle0",
        "kRightHandMiddle1",
    ]

z1_cameras = [
        "cam_high",
        "cam_left_wrist",
        "cam_right_wrist",
    ]
g1_cameras = [
        "cam_left_high",
        "cam_right_high",
        "cam_left_wrist",
        "cam_right_wrist",
    ]
def create_empty_dataset(
    repo_id: str,
    robot_type: str,
    mode: Literal["video", "image"] = "video",
    *,
    has_velocity: bool = False,
    has_effort: bool = False,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
) -> LeRobotDataset:
    motors = g1_motors_with_gripper
    cameras = g1_cameras

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        },
        "action": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        },
    }

    if has_velocity:
        features["observation.velocity"] = {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        }

    if has_effort:
        features["observation.effort"] = {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        }

    for cam in cameras:
        features[f"observation.images.{cam}"] = {
            "dtype": mode,
            "shape": (3, 480, 640),
            "names": [
                "channels",
                "height",
                "width",
            ],
        }

    if Path(LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(LEROBOT_HOME / repo_id)

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=30,
        robot_type=robot_type,
        features=features,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
    )


def get_all_json(data_dir):
    task_paths = []
    episode_paths = []

    for task_path in glob.glob(os.path.join(data_dir, '*')):
        if os.path.isdir(task_path):
            episodes = glob.glob(os.path.join(task_path, '*'))
            if episodes:
                task_paths.append(task_path)
                episode_paths.extend(episodes)

    return task_paths, episode_paths

json_file = 'data.json'

def extract_data(episode_data, key, parts):
    result = []
    for sample_data in episode_data['data']:
        data_array = np.array([], dtype=np.float32)
        for part in parts:
            if part in sample_data[key] and sample_data[key][part] is not None:
                qpos = np.array(sample_data[key][part]['qpos'], dtype=np.float32)
                data_array = np.concatenate([data_array, qpos])
        result.append(data_array)
    return np.array(result)

def get_actions_data(episode_data):
    parts = ['left_arm', 'right_arm', 'left_hand', 'right_hand']
    return extract_data(episode_data, 'actions', parts)

def get_states_data(episode_data):
    parts = ['left_arm', 'right_arm', 'left_hand', 'right_hand']
    return extract_data(episode_data, 'states', parts)


def get_cameras(json_data):
    # ignore depth channel, not currently handled
    # TODO(rcadene): add depth
    keys = json_data["data"][0]['colors'].keys()
    rgb_cameras = [key for key in keys if "depth" not in key]  # noqa: SIM118
    return rgb_cameras


def get_images_data(ep_path, episode_data):
    images = {}
    # Add the camera_to_image_key as required
    camera_to_image_key = {'color_0': 'cam_left_high', 'color_1':'cam_right_high', 'color_2': 'cam_left_wrist' ,'color_3': 'cam_right_wrist'}
    cameras = get_cameras(episode_data)

    for camera in cameras:
        image_key = camera_to_image_key.get(camera)
        if image_key is None:
            continue
        
        images.setdefault(image_key, [])

        for sample_data in episode_data['data']:
            image_path = os.path.join(ep_path, sample_data['colors'].get(camera, ""))
            if not os.path.exists(image_path):
                print(f"Warning: Image path does not exist: {image_path}")
                continue

            image = cv2.imread(image_path)
            if image is None:
                print(f"Warning: Failed to read image at {image_path}")
                continue
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            images[image_key].append(image_rgb)
    
    return images


def populate_dataset(
    dataset: LeRobotDataset,
    raw_dir: list[Path],
    task: str,
    episodes: list[int] | None = None,
) -> LeRobotDataset:


    task_paths, episode_paths = get_all_json(raw_dir)
    print(f"Found {len(task_paths)} tasks and {len(episode_paths)} episodes.")
    
    episode_paths = sorted(episode_paths, key=lambda path: int(re.search(r'(\d+)$', path).group(1)) if re.search(r'(\d+)$', path) else 0)

    if episodes is None:
        episodes = range(len(episode_paths))

    for ep_idx in tqdm.tqdm(episodes):
        ep_path = episode_paths[ep_idx]
        json_path = os.path.join(ep_path, json_file)

        with open(json_path, 'r', encoding='utf-8') as jsonf:
            episode_data = json.load(jsonf)

            action = torch.from_numpy(get_actions_data(episode_data))
            state = torch.from_numpy(get_states_data(episode_data))
            imgs_per_cam = get_images_data(ep_path, episode_data)

            # imgs_per_cam, state, action, velocity, effort = load_raw_episode_data(ep_path)
            num_frames = action.shape[0]

            for i in range(num_frames):
                frame = {
                    "observation.state": state[i],
                    "action": action[i],
                }

                for camera, img_array in imgs_per_cam.items():
                    frame[f"observation.images.{camera}"] = img_array[i]

                dataset.add_frame(frame)

            dataset.save_episode(task=task, encode_videos=True)

    return dataset


def port_aloha(
    raw_dir: Path,
    repo_id: str,
    raw_repo_id: str | None = None,
    task: str = "pour coffee",
    *,
    episodes: list[int] | None = None,
    push_to_hub: bool = True,
    is_mobile: bool = False,
    mode: Literal["video", "image"] = "video",
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
):
    robot_type = 'Unitree_Z1_Dual'
    task000 = "pour coffee"

    if (LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(LEROBOT_HOME / repo_id)


    dataset = create_empty_dataset(
        repo_id,
        robot_type=robot_type,
        mode=mode,
        has_effort=False,
        has_velocity=False,
        dataset_config=dataset_config,
    )
    dataset = populate_dataset(
        dataset,
        raw_dir,
        task=task000,
        episodes=episodes,
    )
    dataset.consolidate()

    # task = "Henry-Ellis/Z1_DualArm_PourCoffee"
    # root_path = "/home/unitree/datasets/z1/Henry-Ellis/Z1_DualArm_PourCoffee"

    # # We can have a look and fetch its metadata to know more about it:
    # dataset = LeRobotDataset(repo_id = task, root = root_path, local_files_only=True)

    # if push_to_hub:
    #     dataset.push_to_hub()


if __name__ == "__main__":
    tyro.cli(port_aloha)
