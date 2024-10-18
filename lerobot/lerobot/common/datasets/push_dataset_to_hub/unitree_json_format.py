"""
Contains utilities to process raw data format of HDF5 files like in: https://github.com/tonyzhaozh/act
"""

import re
import shutil
from pathlib import Path
import numpy as np

import torch
import tqdm
from datasets import Dataset, Features, Image, Sequence, Value
from PIL import Image as PILImage
import glob
import os
import json
import cv2
from lerobot.common.datasets.lerobot_dataset import CODEBASE_VERSION
from lerobot.common.datasets.utils import (
    hf_transform_to_torch,
    calculate_episode_data_index
)
from lerobot.common.datasets.push_dataset_to_hub.utils import (

    concatenate_episodes,
    get_default_encoding,
    save_images_concurrently,
)
from lerobot.common.datasets.video_utils import VideoFrame, encode_video_frames
json_file = 'data.json'

def check_format(raw_dir) -> bool:
    episode_paths = get_all_json(raw_dir)
    episode_paths = sorted(episode_paths, key=lambda path: int(re.search(r'(\d+)$', path).group(1)) if re.search(r'(\d+)$', path) else 0)
    assert len(episode_paths) != 0
    
    for json_path in episode_paths:
        json_path = os.path.join(json_path, json_file)
        with open(json_path, 'r', encoding='utf-8') as jsonf:

            episode_data = json.load(jsonf)
            for sample_data in episode_data['data']:
                assert "actions" in sample_data
                assert "states" in sample_data
                assert "colors" in sample_data
                assert len(sample_data["colors"]) != 0
                assert len(sample_data["states"]) != 0
                assert len(sample_data["actions"]) != 0

def get_cameras(json_data):
    # ignore depth channel, not currently handled
    # TODO(rcadene): add depth
    keys = json_data["data"][0]['colors'].keys()
    rgb_cameras = [key for key in keys if "depth" not in key]  # noqa: SIM118
    return rgb_cameras

# def get_all_json(data_dir):
#     task_paths = []
#     episode_paths = []
#     for task_path in glob.glob(os.path.join(data_dir, '*')):
#         temepisode_paths = glob.glob(os.path.join(task_path, '*'))
#         if os.path.isdir(task_path) and len(temepisode_paths) > 0:
#             task_paths.append(task_path)
#             episode_paths.extend(temepisode_paths)
#     return task_paths,episode_paths

def get_all_json(data_dir):
    episode_paths = []
    temepisode_paths = glob.glob(os.path.join(data_dir, '*'))
    if os.path.isdir(data_dir) and len(temepisode_paths) > 0:
        episode_paths.extend(temepisode_paths)
    return episode_paths

# def get_all_json(data_dir):
#     task_paths = []
#     episode_paths = []

#     for task_path in glob.glob(os.path.join(data_dir, '*')):
#         if os.path.isdir(task_path):
#             episodes = glob.glob(os.path.join(task_path, '*'))
#             if episodes:
#                 task_paths.append(task_path)
#                 episode_paths.extend(episodes)

#     return task_paths, episode_paths


# def get_arm_qpos(data, arm_name, key):
#     arm_data = data.get(key, {}).get(arm_name)
#     if arm_data is not None and 'qpos' in arm_data:
#         return np.array(arm_data['qpos'], dtype=np.float32)
#     return np.array([], dtype=np.float32)

# def get_data(json_path, json_file, key):
#     results = []
#     json_path = os.path.join(json_path, json_file)
    
#     with open(json_path, 'r', encoding='utf-8') as jsonf:
#         episode_data = json.load(jsonf)
        
#         for sample_data in episode_data['data']:
#             result = np.concatenate([
#                 get_arm_qpos(sample_data, 'left_arm', key),
#                 get_arm_qpos(sample_data, 'right_arm', key)
#             ])
#             results.append(result)
    
#     return np.array(results)
# states = get_data('path_to_json', 'file.json', 'states')
# actions = get_data('path_to_json', 'file.json', 'actions')

# def get_images_data(path, json_file):
#     """获取图像数据"""
#     images = {}
#     json_file_path = os.path.join(path, json_file)
    
#     with open(json_file_path, 'r', encoding='utf-8') as jsonf:
#         episode_data = json.load(jsonf)
#         cameras = get_cameras(episode_data)
        
#         for camera in cameras:
#             image_key = 'top' if camera == 'color_0' else 'wrist' if camera == 'color_1' else None
#             if image_key not in images:
#                 images[image_key] = []
            
#             for sample_data in episode_data['data']:
#                 image_path = os.path.join(path, sample_data['colors'][camera])
#                 image = cv2.imread(image_path)
#                 image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#                 images[image_key].append(image_rgb)
    
#     return images

def get_actions_data(json_path):
    actions=[]
    json_path = os.path.join(json_path, json_file)
    with open(json_path, 'r', encoding='utf-8') as jsonf:
        episode_data = json.load(jsonf)
        for sample_data in episode_data['data']:
            action = np.array([], dtype=np.float32)
            if 'left_arm' in sample_data['actions'] and sample_data['actions']['left_arm'] is not None:
                qpos = np.array(sample_data['actions']['left_arm']['qpos']).astype(np.float32)
                action = np.concatenate([action, qpos])
            if 'right_arm' in sample_data['actions'] and sample_data['actions']['right_arm'] is not None:
                qpos = np.array(sample_data['actions']['right_arm']['qpos']).astype(np.float32)
                action = np.concatenate([action, qpos])
            if 'left_hand' in sample_data['actions'] and sample_data['actions']['left_hand'] is not None:
                qpos = np.array(sample_data['actions']['left_hand']['qpos']).astype(np.float32)
                action = np.concatenate([action, qpos])
            if 'right_hand' in sample_data['actions'] and sample_data['actions']['right_hand'] is not None:
                qpos = np.array(sample_data['actions']['right_hand']['qpos']).astype(np.float32)
                action = np.concatenate([action, qpos])
            actions.append(action)                          # L, D                         
    actions = np.array(actions)
    return actions

def get_states_data(json_path):

    states = []
    json_path = os.path.join(json_path, json_file)
    with open(json_path, 'r', encoding='utf-8') as jsonf:
        episode_data = json.load(jsonf)
        for sample_data in episode_data['data']:
            state = np.array([], dtype=np.float32)
            if 'left_arm' in sample_data['states'] and sample_data['states']['left_arm'] is not None:
                qpos = np.array(sample_data['states']['left_arm']['qpos']).astype(np.float32)
                state = np.concatenate([state, qpos])
            if 'right_arm' in sample_data['states'] and sample_data['states']['right_arm'] is not None:
                qpos = np.array(sample_data['states']['right_arm']['qpos']).astype(np.float32)
                state = np.concatenate([state, qpos])
            if 'left_hand' in sample_data['states'] and sample_data['states']['left_hand'] is not None:
                qpos = np.array(sample_data['states']['left_hand']['qpos']).astype(np.float32)
                state = np.concatenate([state, qpos])
            if 'right_hand' in sample_data['states'] and sample_data['states']['right_hand'] is not None:
                qpos = np.array(sample_data['states']['right_hand']['qpos']).astype(np.float32)
                state = np.concatenate([state, qpos])
            states.append(state)        # L, D         
                
    states = np.array(states)
    return states

def get_images_data(path):
    images = {}
    json_path = os.path.join(path, json_file)
    with open(json_path, 'r', encoding='utf-8') as jsonf:
        episode_data = json.load(jsonf)
        cameras = get_cameras(episode_data)
        for camera in cameras:
            if camera=='color_0':
                image_key='top'
            elif camera=='color_1':
                image_key='wrist'
            if image_key not in images:
                images[image_key]=[]
            for sample_data in episode_data['data']:
                image_path = os.path.join(path,sample_data['colors'][camera])
                image = cv2.imread(image_path)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                images[image_key].append(image_rgb)
    return images


def load_from_raw(
    raw_dir: Path,
    videos_dir: Path,
    fps: int,
    video: bool,
    episodes: list[int] | None = None,
    encoding: dict | None = None,
):
    episode_paths = get_all_json(raw_dir)
       
    episode_paths = sorted(episode_paths, key=lambda path: int(re.search(r'(\d+)$', path).group(1)) if re.search(r'(\d+)$', path) else 0)
    num_episodes = len(episode_paths)
    
    ep_dicts = [] 
    ep_ids = episodes if episodes else range(num_episodes)
    for ep_idx in tqdm.tqdm(ep_ids):
        ep_path = episode_paths[ep_idx]

        action = torch.from_numpy(get_actions_data(ep_path))
        state = torch.from_numpy(get_states_data(ep_path))
        
        num_frames = action.shape[0]
        # last step of demonstration is considered done
        done = torch.zeros(num_frames, dtype=torch.bool)
        done[-1] = True

        ep_dict={}

        image_dic = get_images_data(ep_path)
        for camera, imgs_array in image_dic.items():
            assert num_frames==len(imgs_array)
    
            img_key = f"observation.images.{camera}"
            if video:
                # save png images in temporary directory
                tmp_imgs_dir = videos_dir / "tmp_images"
                save_images_concurrently(imgs_array, tmp_imgs_dir)

                # encode images to a mp4 video
                fname = f"{img_key}_episode_{ep_idx:06d}.mp4"
                video_path = os.path.join(videos_dir, fname)
                encode_video_frames(tmp_imgs_dir, video_path, fps, vcodec='libx264')
                # encode_video_frames(tmp_imgs_dir, video_path, fps, **(encoding or {}))

                # clean temporary images directory
                shutil.rmtree(tmp_imgs_dir)
                
                # store the reference to the video frame
                ep_dict[img_key] = [
                    {"path": f"videos/{fname}", "timestamp": i / fps} for i in range(num_frames)
                ]
            else:
                ep_dict[img_key] = [PILImage.fromarray(x) for x in imgs_array]
        
        ep_dict["observation.state"] = state
        ep_dict["action"] = action
        ep_dict["episode_index"] = torch.tensor([ep_idx] * num_frames)
        ep_dict["frame_index"] = torch.arange(0, num_frames, 1)
        ep_dict["timestamp"] = torch.arange(0, num_frames, 1) / fps
        ep_dict["next.done"] = done

        assert isinstance(ep_idx, int)
        ep_dicts.append(ep_dict)

    data_dict = concatenate_episodes(ep_dicts)
    total_frames = data_dict["frame_index"].shape[0]
    data_dict["index"] = torch.arange(0, total_frames, 1)
    return data_dict

def to_hf_dataset(data_dict, video) -> Dataset:
    features = {}

    keys = [key for key in data_dict if "observation.images." in key]
    for key in keys:
        if video:
            features[key] = VideoFrame()
        else:
            features[key] = Image()

    features["observation.state"] = Sequence(
        length=data_dict["observation.state"].shape[1], feature=Value(dtype="float32", id=None)
    )
    features["action"] = Sequence(
        length=data_dict["action"].shape[1], feature=Value(dtype="float32", id=None)
    )
    features["episode_index"] = Value(dtype="int64", id=None)
    features["frame_index"] = Value(dtype="int64", id=None)
    features["timestamp"] = Value(dtype="float32", id=None)
    features["next.done"] = Value(dtype="bool", id=None)
    features["index"] = Value(dtype="int64", id=None)

    hf_dataset = Dataset.from_dict(data_dict, features=Features(features))
    hf_dataset.set_transform(hf_transform_to_torch)
    return hf_dataset

def from_raw_to_lerobot_format(
    raw_dir: Path,
    videos_dir: Path,
    fps: int | None = None,
    video: bool = True,
    episodes: list[int] | None = None,
    encoding: dict | None = None,
):

    if fps is None:
        fps = 30
    check_format(raw_dir)
    data_dict = load_from_raw(raw_dir, videos_dir, fps, video, episodes, encoding)
    hf_dataset = to_hf_dataset(data_dict, video)
    episode_data_index = calculate_episode_data_index(hf_dataset)

    info = {
        "codebase_version": CODEBASE_VERSION,
        "fps": fps,
        "video": video,
    }
    if video:
        info["encoding"] = get_default_encoding()
    return hf_dataset, episode_data_index, info

