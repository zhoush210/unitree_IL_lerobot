#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import cv2
import json
import logging
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from pprint import pformat
from copy import copy
import threading
import numpy as np
import torch
from torch import Tensor, nn

from lerobot.common.envs.utils import preprocess_observation
from lerobot.common.policies.factory import make_policy
from lerobot.common.policies.utils import get_device_from_parameters
from lerobot.common.utils.utils import (
    get_safe_torch_device,
    init_logging,
)
from lerobot.configs import parser
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


from unitree_lerobot.eval_robot.eval_g1.image_server.image_client import ImageClient
from unitree_lerobot.eval_robot.eval_g1.robot_control.robot_arm import G1_29_ArmController
from unitree_lerobot.eval_robot.eval_g1.robot_control.robot_hand_unitree import Dex3_1_Controller, Gripper_Controller
from unitree_lerobot.eval_robot.eval_g1.eval_real_config import EvalRealConfig


from multiprocessing import Process, shared_memory, Array
from multiprocessing import shared_memory, Array, Lock

def get_image_processed(cam, img_size=[640, 480]):
    # realsense return cv2 image, BGR format
    curr_images = []
    color_img  = cam
    color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
    color_img = cv2.resize(color_img, img_size)         # w, h
    curr_images.append(color_img)
    color_img = np.stack(curr_images, axis=0)
    return color_img


# copy from lerobot.common.robot_devices.control_utils import predict_action
def predict_action(observation, policy, device, use_amp):
    observation = copy(observation)
    with (
        torch.inference_mode(),
        torch.autocast(device_type=device.type) if device.type == "cuda" and use_amp else nullcontext(),
    ):
        # Convert to pytorch format: channel first and float32 in [0,1] with batch dimension
        for name in observation:
            if "image" in name:
                observation[name] = observation[name].type(torch.float32) / 255
                observation[name] = observation[name].permute(2, 0, 1).contiguous()
            observation[name] = observation[name].unsqueeze(0)
            observation[name] = observation[name].to(device)

        # Compute the next action with the policy
        # based on the current observation
        action = policy.select_action(observation)

        # Remove batch dimension
        action = action.squeeze(0)

        # Move to cpu, if not already the case
        action = action.to("cpu")

    return action


def eval_policy(
    policy: torch.nn.Module,
    dataset: LeRobotDataset,
) -> dict:
    
    assert isinstance(policy, nn.Module), "Policy must be a PyTorch nn module."
    device = get_device_from_parameters(policy)

    # Reset the policy and environments.
    policy.reset()

    #Shared memory for storing images. The experiment uses a stereo camera, with each image being 480x640x3 (hxwx3).
    img_config = {
        'fps': 30,
        'head_camera_type': 'opencv',
        'head_camera_image_shape': [480, 1280],  # Head camera resolution
        'head_camera_id_numbers': [0],
        'wrist_camera_type': 'opencv',
        'wrist_camera_image_shape': [480, 640],  # Wrist camera resolution
        'wrist_camera_id_numbers': [2, 4],
    }
    ASPECT_RATIO_THRESHOLD = 2.0 # If the aspect ratio exceeds this value, it is considered binocular
    if len(img_config['head_camera_id_numbers']) > 1 or (img_config['head_camera_image_shape'][1] / img_config['head_camera_image_shape'][0] > ASPECT_RATIO_THRESHOLD):
        BINOCULAR = True
    else:
        BINOCULAR = False
    if 'wrist_camera_type' in img_config:
        WRIST = True
    else:
        WRIST = False
    
    if BINOCULAR and not (img_config['head_camera_image_shape'][1] / img_config['head_camera_image_shape'][0] > ASPECT_RATIO_THRESHOLD):
        tv_img_shape = (img_config['head_camera_image_shape'][0], img_config['head_camera_image_shape'][1] * 2, 3)
    else:
        tv_img_shape = (img_config['head_camera_image_shape'][0], img_config['head_camera_image_shape'][1], 3)

    tv_img_shm = shared_memory.SharedMemory(create = True, size = np.prod(tv_img_shape) * np.uint8().itemsize)
    tv_img_array = np.ndarray(tv_img_shape, dtype = np.uint8, buffer = tv_img_shm.buf)

    if WRIST:
        wrist_img_shape = (img_config['wrist_camera_image_shape'][0], img_config['wrist_camera_image_shape'][1] * 2, 3)
        wrist_img_shm = shared_memory.SharedMemory(create = True, size = np.prod(wrist_img_shape) * np.uint8().itemsize)
        wrist_img_array = np.ndarray(wrist_img_shape, dtype = np.uint8, buffer = wrist_img_shm.buf)
        img_client = ImageClient(tv_img_shape = tv_img_shape, tv_img_shm_name = tv_img_shm.name, 
                                 wrist_img_shape = wrist_img_shape, wrist_img_shm_name = wrist_img_shm.name)
    else:
        img_client = ImageClient(tv_img_shape = tv_img_shape, tv_img_shm_name = tv_img_shm.name)

    image_receive_thread = threading.Thread(target = img_client.receive_process, daemon = True)
    image_receive_thread.daemon = True
    image_receive_thread.start()

    robot_config = {
        'arm_type': 'g1',
        'hand_type': "dex3",
    }

    # init pose
    from_idx = dataset.episode_data_index["from"][0].item()
    step = dataset[from_idx]

    # arm
    arm_ctrl = G1_29_ArmController()
    init_left_arm_pose = step['observation.state'][:13]
    # hand
    if robot_config['hand_type'] == "dex3":
        left_hand_array = Array('d', 7, lock = True)          # [input]
        right_hand_array = Array('d', 7, lock = True)         # [input]
        dual_hand_data_lock = Lock()
        dual_hand_state_array = Array('d', 14, lock = False)  # [output] current left, right hand state(14) data.
        dual_hand_action_array = Array('d', 14, lock = False) # [output] current left, right hand action(14) data.
        hand_ctrl = Dex3_1_Controller(left_hand_array, right_hand_array, dual_hand_data_lock, dual_hand_state_array, dual_hand_action_array)
        init_left_hand_pose = step['observation.state'][13:19]
        init_right_hand_pose = step['observation.state'][19:]

    elif robot_config['hand_type'] == "gripper":
        left_hand_array = Array('d', 1, lock=True)             # [input]
        right_hand_array = Array('d', 1, lock=True)            # [input]
        dual_gripper_data_lock = Lock()
        dual_gripper_state_array = Array('d', 2, lock=False)   # current left, right gripper state(2) data.
        dual_gripper_action_array = Array('d', 2, lock=False)  # current left, right gripper action(2) data.
        gripper_ctrl = Gripper_Controller(left_hand_array, right_hand_array, dual_gripper_data_lock, dual_gripper_state_array, dual_gripper_action_array)
        init_left_hand_pose = step['observation.state'][13]
        init_right_hand_pose = step['observation.state'][14]
    else:
        pass

    #===============init robot=====================
    user_input = input("Please enter the start signal (enter 's' to start the subsequent program):")
    if user_input.lower() == 's':

        # "The initial positions of the robot's arm and fingers take the initial positions during data recording."
        print(f"init robot pose")
        arm_ctrl.ctrl_dual_arm(init_left_arm_pose, np.zeros(14))
        left_hand_array[:] = init_left_hand_pose
        right_hand_array[:] = init_right_hand_pose

        print(f"wait robot to pose")
        time.sleep(2)

        frequency = 15.0  # 15 Hz

        while True:

            observation = {}
            img_dic = dict()

            current_tv_image = tv_img_array.copy()
            # wrist image
            if WRIST:
                current_wrist_image = wrist_img_array.copy()
            #Get the current image.
            left_top_camera = current_tv_image[:, :tv_img_shape[1]//2]
            right_top_camera = current_tv_image[:, tv_img_shape[1]//2:]
            if WRIST:
                left_wrist_camera = current_wrist_image[:, :wrist_img_shape[1]//2]
                right_wrist_camera = current_wrist_image[:, wrist_img_shape[1]//2:]

            img_dic['cam_left_high'] = get_image_processed(left_top_camera)
            img_dic['cam_right_high'] = get_image_processed(right_top_camera)
            img_dic['cam_left_wrist'] = get_image_processed(left_wrist_camera)
            img_dic['cam_right_wrist'] = get_image_processed(right_wrist_camera)

            # get current state data.
            current_lr_arm_q  = arm_ctrl.get_current_dual_arm_q()
            # dex hand or gripper
            if robot_config['hand_type'] == "dex3":
                with dual_hand_data_lock:
                    left_hand_state = dual_hand_state_array[:7]
                    right_hand_state = dual_hand_state_array[-7:]
            elif robot_config['hand_type'] == "gripper":
                with dual_gripper_data_lock:
                    left_hand_state = [dual_gripper_state_array[1]]
                    right_hand_state = [dual_gripper_state_array[0]]
            
            robot_state = np.concatenate((current_lr_arm_q, left_hand_state, right_hand_state), axis=0)
            observation["pixels"] = img_dic
            observation["agent_pos"] = robot_state
            observation = preprocess_observation(observation)
            observation['observation.state'] = observation['observation.state'].unsqueeze(0)

            # Numpy array to tensor and changing dictionary keys to LeRobot policy format.
            observation = preprocess_observation(observation)
            observation = {
                key: observation[key].to(device, non_blocking=device.type == "cuda") for key in observation
            }

            action = predict_action(
                observation, policy, get_safe_torch_device(policy.config.device), policy.config.use_amp
            )


            print(f"qpose:{np.round(action / np.pi * 180, 1)}")

            arm_ctrl.ctrl_dual_arm(action[:13], np.zeros(14))
            if robot_config['hand_type'] == "dex3":
                left_hand_array[:] = action[13:19]
                right_hand_array[:] = action[19:]
            elif robot_config['hand_type'] == "gripper":
                left_hand_array[:] = action[13]
                right_hand_array[:] = action[14]
        
            time.sleep(1/frequency)


@parser.wrap()
def eval_main(cfg: EvalRealConfig):
    logging.info(pformat(asdict(cfg)))

    # Check device is available
    device = get_safe_torch_device(cfg.policy.device, log=True)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    logging.info("Making policy.")

    dataset = LeRobotDataset(repo_id = cfg.repo_id)

    policy = make_policy(
        cfg=cfg.policy,
        ds_meta=dataset.meta
    )
    policy.eval()

    with torch.no_grad(), torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext():
        info = eval_policy(policy, dataset)
    print(info["aggregated"])

    logging.info("End of eval")


if __name__ == "__main__":
    init_logging()
    eval_main()

