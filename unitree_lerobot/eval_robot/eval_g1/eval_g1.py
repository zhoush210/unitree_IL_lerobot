''''
Refer to:   lerobot/lerobot/scripts/eval.py
            lerobot/lerobot/scripts/econtrol_robot.py
            lerobot/common/robot_devices/control_utils.py
'''

import time
import torch
import logging
import threading
import numpy as np
from copy import copy
from pprint import pformat
from dataclasses import asdict
from torch import nn
from contextlib import nullcontext
from multiprocessing import shared_memory, Array, Lock

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
import test_act
import os
import cv2
from datetime import datetime
import sys
import select
import tty

# copy from lerobot.common.robot_devices.control_utils import predict_action
def predict_action(observation, policy, device, use_amp):
    observation = copy(observation)
    with (
        torch.inference_mode(),
        torch.autocast(device_type=device.type) if device.type == "cuda" and use_amp else nullcontext(),
    ):
        # Convert to pytorch format: channel first and float32 in [0,1] with batch dimension
        for name in observation:
            if "images" in name:
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

def save_episode_video(frames, rollout_id, time_dir, base_dir="./eval_videos", fps=30):
    """
    将本次 rollout 的图片序列保存为 mp4 视频
    frames: list[np.ndarray]
    rollout_id: 当前 rollout 的编号
    base_dir: 保存目录的父路径
    fps: 视频帧率
    """
    if not frames:
        print(f"[Warning] rollout {rollout_id} 没有采集到任何帧，视频未保存。")
        return

    save_dir = os.path.join(base_dir, time_dir)
    os.makedirs(save_dir, exist_ok=True)

    height, width = frames[0].shape[:2]
    video_path = os.path.join(save_dir, f"rollout_{rollout_id}.mp4")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    for frame in frames:
        if frame.dtype != 'uint8':
            frame = (frame * 255).astype('uint8')
        if frame.shape[2] == 3:
            out.write(frame)
        else:
            print("[Warning] 跳过一帧:不是3通道图像")

    out.release()
    print(f"[Info] Rollout {rollout_id} 视频已保存到 {video_path}")

def get_key_nonblocking():
    """非阻塞读取一个按键，没有输入则返回 None"""
    dr, _, _ = select.select([sys.stdin], [], [], 0)
    if dr:
        return sys.stdin.read(1)  # 读取一个字符
    return None

def eval_policy(
    policy: torch.nn.Module,
    dataset: LeRobotDataset,
):
    
    assert isinstance(policy, nn.Module), "Policy must be a PyTorch nn module."
    device = get_device_from_parameters(policy)

    # Reset the policy and environments.
    policy.reset()

    num_rollouts = 50
    time_dir = datetime.now().strftime("%Y%m%d_%H%M%S")
    for rollout_id in range(num_rollouts):
        robot_config = {
            'arm_type': 'g1',
            'hand_type': "dex3",
        }

    
        # init custom
        test_act.ChannelFactoryInitialize(0)
        custom = test_act.Custom()
        custom.Init()
        while not custom.first_update_low_state:
            time.sleep(0.02)
        # init pose
        from_idx = dataset.episode_data_index["from"][0].item()
        step = dataset[from_idx]
        to_idx = dataset.episode_data_index["to"][0].item()

        # arm
        # arm_ctrl = G1_29_ArmController()
        init_left_arm_pose = step['observation.state'][:14].cpu().numpy()

        # hand
        if robot_config['hand_type'] == "dex3":
            left_hand_array = Array('d', 7, lock = True)          # [input]
            right_hand_array = Array('d', 7, lock = True)         # [input]
            dual_hand_data_lock = Lock()
            dual_hand_state_array = Array('d', 14, lock = False)  # [output] current left, right hand state(14) data.
            dual_hand_action_array = Array('d', 14, lock = False) # [output] current left, right hand action(14) data.
            hand_ctrl = Dex3_1_Controller(
            right_hand_state_array  = right_hand_array,
            left_hand_state_array  = left_hand_array,
            dual_hand_action_array = dual_hand_action_array,
            fps        = 100.0,     # control frequency
            Unit_Test  = False
        )
            init_left_hand_pose = step['observation.state'][14:21].cpu().numpy()
            init_right_hand_pose = step['observation.state'][21:].cpu().numpy()

        elif robot_config['hand_type'] == "gripper":
            left_hand_array = Array('d', 1, lock=True)             # [input]
            right_hand_array = Array('d', 1, lock=True)            # [input]
            dual_gripper_data_lock = Lock()
            dual_gripper_state_array = Array('d', 2, lock=False)   # current left, right gripper state(2) data.
            dual_gripper_action_array = Array('d', 2, lock=False)  # current left, right gripper action(2) data.
            gripper_ctrl = Gripper_Controller(left_hand_array, right_hand_array, dual_gripper_data_lock, dual_gripper_state_array, dual_gripper_action_array)
            init_left_hand_pose = step['observation.state'][14].cpu().numpy()
            init_right_hand_pose = step['observation.state'][15].cpu().numpy()
        else:
            pass

        #===============init robot=====================
        # "The initial positions of the robot's arm and fingers take the initial positions during data recording."
        print("init robot pose")
        # arm_ctrl.ctrl_dual_arm(init_left_arm_pose, np.zeros(14))
        left_hand_array[:] = init_left_hand_pose
        right_hand_array[:] = init_right_hand_pose

        # init_left_arm = np.zeros(7)
        # init_right_arm = np.array([0,0,-0.3,-0.2,0,0,0])  # Right arm joint positions
        init_left_arm = np.array([-0.1218879421015483, 0.1936419989831646, 0.17425482576988505, 0.7443733704658254, -0.14442309218100005, -0.8512422138131861, -0.3431697949914991])
        init_right_arm = np.array([-0.5557356425795662, -0.24198717407556442, 0.16197737375770552, 0.9128680627438711, -0.3724260954466061, -0.5365253098423672, -0.1674598238324641])
        custom.init_armpos(init_left_arm, init_right_arm)
        print("wait robot to pose")
        time.sleep(1)
        # init_left_hand = np.zeros(7)
        # init_right_hand = np.zeros(7)
        init_left_hand = np.array([-0.9206724696209045, 0.5992037963321957, 0.3267422976110949, -0.022891257938381884, -0.09687889165773123, -0.021911120791004147, -0.059875848573434705])
        init_right_hand = np.array([-0.8140116822994757, -0.7375885831248487, -0.08884019141664193, 0.21725622993157398, 0.6336620483743846, 0.30790616123599096, 0.4807725104665246])
        hand_ctrl.ctrl_dual_hand(init_left_hand,  init_right_hand)
        time.sleep(1)

        frequency = 50.0

        tty.setcbreak(sys.stdin.fileno())
        print("按回车键开始推理，按 's' 键结束推理")
        input()

        # image
        img_config = {
            'fps': 30,
            'head_camera_type': 'opencv',
            'head_camera_image_shape': [720, 1280],  # Head camera resolution
            'head_camera_id_numbers': [0],
            # 'wrist_camera_type': 'opencv',
            # 'wrist_camera_image_shape': [720, 640],  # Wrist camera resolution
            # 'wrist_camera_id_numbers': [2, 4],
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

        frames = []
        max_step = 500
        for i in range(max_step):
            print("step:", i)
            key = get_key_nonblocking()
            if key == 's':
                print("检测到按键 's'，退出循环")
                break
            # Get images
            current_tv_image = tv_img_array.copy()
            frames.append(current_tv_image.copy())
            current_wrist_image = wrist_img_array.copy() if WRIST else None

            # Assign image data
            left_top_camera = current_tv_image[:, :tv_img_shape[1] // 2] if BINOCULAR else current_tv_image
            right_top_camera = current_tv_image[:, tv_img_shape[1] // 2:] if BINOCULAR else None
            left_wrist_camera, right_wrist_camera = (
                (current_wrist_image[:, :wrist_img_shape[1] // 2], current_wrist_image[:, wrist_img_shape[1] // 2:])
                if WRIST else (None, None)
            )

            observation = {
                "observation.images.cam_left_high": torch.from_numpy(left_top_camera),
                # "observation.images.cam_right_high": torch.from_numpy(right_top_camera) if BINOCULAR else None,
                # "observation.images.cam_left_wrist": torch.from_numpy(left_wrist_camera) if WRIST else None,
                # "observation.images.cam_right_wrist": torch.from_numpy(right_wrist_camera) if WRIST else None,
            }

            # get current state data.
            # current_lr_arm_q  = arm_ctrl.get_current_dual_arm_q()
            current_lr_arm_q = [custom.low_state.motor_state[j].q for j in custom.arm_joints][0:14]
            
            # dex hand or gripper
            if robot_config['hand_type'] == "dex3":
                with dual_hand_data_lock:
                    left_hand_state = dual_hand_state_array[:7]
                    right_hand_state = dual_hand_state_array[-7:]
            elif robot_config['hand_type'] == "gripper":
                with dual_gripper_data_lock:
                    left_hand_state = [dual_gripper_state_array[1]]
                    right_hand_state = [dual_gripper_state_array[0]]
            
            observation["observation.state"] = torch.from_numpy(np.concatenate((current_lr_arm_q, left_hand_state, right_hand_state), axis=0)).float()

            observation = {
                key: observation[key].to(device, non_blocking=device.type == "cuda") for key in observation
            }

            action = predict_action(
                observation, policy, get_safe_torch_device(policy.config.device), policy.config.use_amp
            )
            action = action.cpu().numpy()
            
            # 将14维动作扩展为28维以控制机器人
            action_28d = expand_14d_to_28d_action(action)
            
            state = observation["observation.state"].cpu().numpy()

            # excute action
            # arm_ctrl.ctrl_dual_arm(action[:14], np.zeros(14))
            target_qpos = np.concatenate([action_28d[:14], np.zeros(3)], axis=0)
            custom.set_arm_pose(target_qpos, enable_sdk=True)
            hand_ctrl.ctrl_dual_hand(action_28d[14:21],  action_28d[21:])

            if robot_config['hand_type'] == "dex3":
                left_hand_array[:] = action_28d[14:21]
                right_hand_array[:] = action_28d[21:]
            elif robot_config['hand_type'] == "gripper":
                left_hand_array[:] = action_28d[14]
                right_hand_array[:] = action_28d[15]
        
            time.sleep(1/frequency)
        save_episode_video(frames, rollout_id, time_dir)
        tv_img_shm.close()
        tv_img_shm.unlink()
        if WRIST:
            wrist_img_shm.close()
            wrist_img_shm.unlink()
        print("End of eval_policy")

# 右手14个自由度对应的原始28维数组索引
RIGHT_HAND_ACTION_INDICES = [7, 8, 9, 10, 11, 12, 13, 14, 22, 23, 24, 25, 26, 27]

def expand_14d_to_28d_action(action_14d: np.ndarray) -> np.ndarray:
    """
    将14维动作扩展为28维动作，用于控制机器人
    
    Args:
        action_14d: 14维动作数组
        
    Returns:
        28维动作数组，右手位置填充14维动作，其他位置填充0
    """
    action_28d = np.zeros(28)
    action_28d[RIGHT_HAND_ACTION_INDICES] = action_14d
    return action_28d

def filter_dataset_stats_for_right_hand(dataset_stats: dict) -> dict:
    """
    过滤数据集统计信息，只保留右手的14个自由度
    
    Args:
        dataset_stats: 原始数据集统计信息
        
    Returns:
        过滤后的统计信息
    """
    filtered_stats = {}
    for key, value in dataset_stats.items():
        if key == "action" and isinstance(value, dict):
            filtered_action_stats = {}
            for stat_type, stat_value in value.items():
                if hasattr(stat_value, 'shape') and len(stat_value.shape) > 0 and stat_value.shape[-1] == 28:
                    # 提取右手的14个自由度统计信息
                    filtered_action_stats[stat_type] = stat_value[..., RIGHT_HAND_ACTION_INDICES]
                elif hasattr(stat_value, '__len__') and len(stat_value) == 28:
                    # 处理list或numpy数组
                    filtered_action_stats[stat_type] = [stat_value[i] for i in RIGHT_HAND_ACTION_INDICES]
                else:
                    filtered_action_stats[stat_type] = stat_value
            filtered_stats[key] = filtered_action_stats
        else:
            filtered_stats[key] = value
    
    return filtered_stats

@parser.wrap()
def eval_main(cfg: EvalRealConfig):
    logging.info(pformat(asdict(cfg)))

    # Check device is available
    device = get_safe_torch_device(cfg.policy.device, log=True)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    logging.info("Making policy.")

    dataset = LeRobotDataset(repo_id = cfg.repo_id)

    # 修改数据集元数据以支持14维动作输出
    if hasattr(dataset.meta, 'features') and 'action' in dataset.meta.features:
        # 创建一个新的元数据对象，手动设置所需的属性
        from copy import deepcopy
        modified_meta = type(dataset.meta)(
            repo_id=dataset.meta.repo_id,
            root=dataset.meta.root,
            revision=dataset.meta.revision
        )
        
        # 复制所有属性
        modified_meta.info = deepcopy(dataset.meta.info)
        modified_meta.tasks = deepcopy(dataset.meta.tasks)
        modified_meta.task_to_task_index = deepcopy(dataset.meta.task_to_task_index)
        modified_meta.episodes = deepcopy(dataset.meta.episodes)
        modified_meta.episodes_stats = deepcopy(dataset.meta.episodes_stats)
        
        # 修改features - 通过修改info字典来设置features
        modified_meta.info['features'] = deepcopy(dataset.meta.info['features'])
        modified_meta.info['features']['action'] = dict(modified_meta.info['features']['action'])
        modified_meta.info['features']['action']['shape'] = (14,)
        logging.info(f"Updated action feature shape to {modified_meta.info['features']['action']['shape']}")
        
        # 过滤数据集统计信息以匹配14维动作
        if hasattr(dataset, 'stats') and dataset.stats:
            modified_meta.stats = filter_dataset_stats_for_right_hand(dataset.stats)
            logging.info("Filtered dataset statistics for right hand actions")
        else:
            modified_meta.stats = deepcopy(dataset.meta.stats)
            if 'action' in modified_meta.stats:
                modified_meta.stats = filter_dataset_stats_for_right_hand(modified_meta.stats)
                logging.info("Filtered existing stats for right hand actions")
    else:
        modified_meta = dataset.meta

    policy = make_policy(
        cfg=cfg.policy,
        ds_meta=modified_meta
    )
    policy.eval()

    with torch.no_grad(), torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext():
        eval_policy(policy, dataset)

    logging.info("End of eval")


if __name__ == "__main__":
    init_logging()
    eval_main()
