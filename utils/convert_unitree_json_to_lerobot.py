"""
Script to convert Aloha hdf5 data to the LeRobot dataset v2.0 format.

Example usage: uv run examples/aloha_real/convert_aloha_data_to_lerobot.py --raw-dir /path/to/raw/data --repo-id <org>/<dataset-name>
"""

import dataclasses
from pathlib import Path
import shutil
from typing import Literal, List, Dict
from lerobot.common.constants import HF_LEROBOT_HOME

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
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


@dataclasses.dataclass(frozen=True)
class RobotConfig:
    motors: List[str]
    cameras: List[str]
    camera_to_image_key:Dict[str, str]
    json_data_name: List[str]


Z1_CONFIG = RobotConfig(
    motors=[
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
    ],
    cameras=[
        "cam_high",
        "cam_left_wrist",
        "cam_right_wrist",
    ],
    camera_to_image_key = {'color_0': 'cam_high', 'color_1': 'cam_left_wrist' ,'color_2': 'cam_right_wrist'},
    json_data_name = ['left_arm', 'right_arm']
)


G1_GRIPPER_CONFIG = RobotConfig(
    motors=[
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
        "kLeftGripper",
        "kRightGripper",
    ],
    cameras=[
        "cam_left_high",
        "cam_right_high",
        "cam_left_wrist",
        "cam_right_wrist",
        ],
    camera_to_image_key = {'color_0': 'cam_left_high', 'color_1':'cam_right_high', 'color_2': 'cam_left_wrist' ,'color_3': 'cam_right_wrist'},
    json_data_name = ['left_arm', 'right_arm']
)


G1_DEX3_CONFIG = RobotConfig(
    motors=[
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
    ],
    cameras=[
        "cam_left_high",
        "cam_right_high",
        "cam_left_wrist",
        "cam_right_wrist",
    ],
    camera_to_image_key = {'color_0': 'cam_left_high', 'color_1':'cam_right_high', 'color_2': 'cam_left_wrist' ,'color_3': 'cam_right_wrist'},
    json_data_name = ['left_arm', 'right_arm', 'left_hand', 'right_hand']
)


ROBOT_CONFIGS = {
    "Unitree_Z1_Dual": Z1_CONFIG,
    "Unitree_G1_Gripper": G1_GRIPPER_CONFIG,
    "Unitree_G1_Dex3": G1_DEX3_CONFIG,
}


def create_empty_dataset(
    repo_id: str,
    robot_type: str,
    mode: Literal["video", "image"] = "video",
    *,
    has_velocity: bool = False,
    has_effort: bool = False,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
) -> LeRobotDataset:
    
    motors = ROBOT_CONFIGS[robot_type].motors
    cameras = ROBOT_CONFIGS[robot_type].cameras

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

    if Path(HF_LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(HF_LEROBOT_HOME / repo_id)

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


def get_all_json(data_dir: str):
    task_paths = []
    episode_paths = []

    for task_path in glob.glob(os.path.join(data_dir, '*')):
        if os.path.isdir(task_path):
            episodes = glob.glob(os.path.join(task_path, '*'))
            if episodes:
                task_paths.append(task_path)
                episode_paths.extend(episodes)

    return task_paths, episode_paths


def extract_data(
        episode_data, 
        key: str,
        parts: list[str]
        ):

    result = []
    for sample_data in episode_data['data']:
        data_array = np.array([], dtype=np.float32)
        for part in parts:
            if part in sample_data[key] and sample_data[key][part] is not None:
                qpos = np.array(sample_data[key][part]['qpos'], dtype=np.float32)
                data_array = np.concatenate([data_array, qpos])
        result.append(data_array)

    return np.array(result)


def get_images_data(
        ep_path: str, 
        episode_data, 
        robot_type: str
        ):
    
    images = {}
    
    # Add the camera_to_image_key as required
    camera_to_image_key = ROBOT_CONFIGS[robot_type].camera_to_image_key
    
    keys = episode_data["data"][0]['colors'].keys()
    cameras = [key for key in keys if "depth" not in key]
    
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
    robot_type: str,
    episodes: list[int] | None = None,
) -> LeRobotDataset:


    task_paths, episode_paths = get_all_json(raw_dir)
    print(f"Found {len(task_paths)} tasks and {len(episode_paths)} episodes.")
    
    episode_paths = sorted(episode_paths, key=lambda path: int(re.search(r'(\d+)$', path).group(1)) if re.search(r'(\d+)$', path) else 0)

    if episodes is None:
        episodes = range(len(episode_paths))

    for ep_idx in tqdm.tqdm(episodes):
        ep_path = episode_paths[ep_idx]
        json_path = os.path.join(ep_path, 'data.json')

        with open(json_path, 'r', encoding='utf-8') as jsonf:
            episode_data = json.load(jsonf)

            action = torch.from_numpy(extract_data(episode_data, 'actions', ROBOT_CONFIGS[robot_type].json_data_name))
            state = torch.from_numpy(extract_data(episode_data, 'states', ROBOT_CONFIGS[robot_type].json_data_name))
            
            imgs_per_cam = get_images_data(ep_path, episode_data, robot_type)

            num_frames = action.shape[0]

            for i in range(num_frames):
                frame = {
                    "observation.state": state[i],
                    "action": action[i],
                    "task": task
                }

                for camera, img_array in imgs_per_cam.items():
                    frame[f"observation.images.{camera}"] = img_array[i]

                dataset.add_frame(frame)

            dataset.save_episode()

    return dataset


def port_dataset(
    raw_dir: Path,
    repo_id: str,
    robot_type: str,        # Unitree_Z1_Dual, Unitree_G1_Gripper, Unitree_G1_Dex3
    task: str,
    *,
    episodes: list[int] | None = None,
    push_to_hub: bool = False,
    mode: Literal["video", "image"] = "video",
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
):

    if (HF_LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(HF_LEROBOT_HOME / repo_id)

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
        robot_type=robot_type,
        task=task,
        episodes=episodes,
    )
    dataset.consolidate()

    if push_to_hub:
        dataset.push_to_hub(upload_large_folder = True)


def local_push_to_hub(
        repo_id: str,
        root_path: Path,):

    dataset = LeRobotDataset(repo_id = repo_id, root = root_path)
    dataset.push_to_hub(upload_large_folder = True)


if __name__ == "__main__":
    tyro.cli(port_dataset)
