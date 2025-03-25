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
"""Evaluate a policy on an environment by running rollouts and computing metrics.

Usage examples:

You want to evaluate a model from the hub (eg: https://huggingface.co/lerobot/diffusion_pusht)
for 10 episodes.

```
python lerobot/scripts/eval.py -p lerobot/diffusion_pusht eval.n_episodes=10
```

OR, you want to evaluate a model checkpoint from the LeRobot training script for 10 episodes.

```
python lerobot/scripts/eval.py \
    -p outputs/train/diffusion_pusht/checkpoints/005000/pretrained_model \
    eval.n_episodes=10
```

Note that in both examples, the repo/folder should contain at least `config.json`, `config.yaml` and
`model.safetensors`.

Note the formatting for providing the number of episodes. Generally, you may provide any number of arguments
with `qualified.parameter.name=value`. In this case, the parameter eval.n_episodes appears as `n_episodes`
nested under `eval` in the `config.yaml` found at
https://huggingface.co/lerobot/diffusion_pusht/tree/main.
"""

import argparse
import json
import logging
import threading
import time
from contextlib import nullcontext
from copy import deepcopy
from datetime import datetime as dt
from pathlib import Path
import numpy as np
import torch
import cv2
import sys
from huggingface_hub import snapshot_download
from huggingface_hub.utils._errors import RepositoryNotFoundError

from huggingface_hub.utils._validators import HFValidationError
from torch import Tensor, nn

from lerobot.common.datasets.factory import make_dataset
from lerobot.common.envs.factory import make_env
from lerobot.common.envs.utils import preprocess_observation
from lerobot.common.logger import log_output_dir
from lerobot.common.policies.factory import make_policy
from lerobot.common.policies.policy_protocol import Policy
from lerobot.common.policies.utils import get_device_from_parameters
from lerobot.common.utils.io_utils import write_video
from lerobot.common.utils.utils import get_safe_torch_device, init_hydra_config, init_logging, set_global_seed


project_root = Path(__file__).resolve().parents[3]
# print(f"project_root:{project_root}")
sys.path.append(str(project_root))
from unitree_utils.image_server.image_client import ImageClient
from unitree_utils.robot_control.robot_arm import G1_29_ArmController
from unitree_utils.robot_control.robot_hand_unitree import Dex3_1_Controller
from multiprocessing import Process, shared_memory, Array

def get_image_processed(cam, img_size=[640, 480]):
    # realsense return cv2 image, BGR format
    curr_images = []
    color_img  = cam
    color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
    color_img = cv2.resize(color_img, img_size)         # w, h
    curr_images.append(color_img)
    color_img = np.stack(curr_images, axis=0)
    return color_img

def eval_policy(
    policy: torch.nn.Module,
) -> dict:
    
    assert isinstance(policy, nn.Module), "Policy must be a PyTorch nn module."
    device = get_device_from_parameters(policy)

    # Reset the policy and environments.
    policy.reset()
    #The is_single_hand variable is used to distinguish between using both hands or a single hand for operation.
    is_single_hand = True
    #Used to distinguish between left and right hand when using only one hand.
    use_left_hand = True  

    g1_arm = G1_29_ArmController()
    tirhand = Dex3_1_Controller()
    # image
    #Shared memory for storing images. The experiment uses a stereo camera, with each image being 480x640x3 (hxwx3).
    image_h = 480
    image_w = 640
    img_shape = (image_h, image_w*2, 3) 
    img_shm = shared_memory.SharedMemory(create=True, size=np.prod(img_shape) * np.uint8().itemsize)
    img_array = np.ndarray(img_shape, dtype=np.uint8, buffer=img_shm.buf)
    img_client = ImageClient(img_shape = img_shape, img_shm_name = img_shm.name)
    image_receive_thread = threading.Thread(target=img_client.receive_process, daemon=True)
    image_receive_thread.start()
    #===============init robot=====================
    user_input = input("Please enter the start signal (enter 's' to start the subsequent program):")
    if user_input.lower() == 's':
        q_pose=np.zeros(14)
        q_tau_ff=np.zeros(14)
        # "The initial positions of the robot's arm and fingers take the initial positions during data recording."
        print(f"init robot pose")
        targetPos = np.array([-0.573512077331543,
                            0.2732987403869629,
                            0.21803689002990723,
                            0.6957042217254639 - 0.12,
                            -0.1663629710674286,
                            -0.2876760959625244,
                            -0.31290051341056824,
                            -0.5136871337890625,
                            -0.3042418956756592,
                            -0.1306476593017578,
                            0.5430612564086914 - 0.12,
                            0.08878898620605469,
                            -0.0726885199546814,
                            0.2417006492614746], dtype=np.float32)
        q_pose = targetPos
        g1_arm.ctrl_dual_arm(q_pose, q_tau_ff)
        left_hand_action = np.array([-0.8095112442970276,
                            1.0347450971603394,
                            0.7259219288825989,
                            0.1899387389421463,
                            -0.013205771334469318,
                            0.187630757689476,
                            -0.010668930597603321], dtype=np.float32)
        
        right_hand_action = np.array([-0.8546233773231506,
                            -1.0273715257644653,
                            -0.6899679899215698,
                            -0.1772225797176361,
                            0.014323404990136623,
                            -0.14210236072540283,
                            0.010851654224097729], dtype=np.float32)
        tirhand.ctrl(left_hand_action,right_hand_action)
        print(f"wait robot to pose")
        time.sleep(5)

        left_arm_action = np.array([-0.573512077331543,
                            0.2732987403869629,
                            0.21803689002990723,
                            0.6957042217254639 - 0.12,
                            -0.1663629710674286,
                            -0.2876760959625244,
                            -0.31290051341056824,], dtype=np.float32)

        right_arm_action = np.array([-0.5136871337890625,
                            -0.3042418956756592,
                            -0.1306476593017578,
                            0.5430612564086914-0.12,
                            0.08878898620605469,
                            -0.0726885199546814,
                            0.2417006492614746], dtype=np.float32)

        
        frequency = 15.0  # 15 Hz
        period = 1.0 / frequency  
        i=0
        next_time = time.time()
        while i in range(100000):
            observation={}
            img_dic=dict()
            # Retrieve the state of the robot's arm and fingers, each as a 14-dimensional array,
            #  with the first 7 dimensions for the left hand and the last 7 for the right hand.
            armstate = g1_arm.get_current_dual_arm_q()
            handstate = tirhand.get_current_dual_hand_q()
            if is_single_hand: #Default to using the left hand; please adjust according to your situation.
                if use_left_hand:
                    leftarmstate = armstate[:7]   
                    lefthandstate =handstate[:7]
                    qpos_data_processed = np.concatenate([leftarmstate,lefthandstate])
                else:
                    rightarmstate = armstate[-7:]   
                    righthandstate =handstate[-7:]
                    qpos_data_processed = np.concatenate([rightarmstate,righthandstate])
            else:
                qpos_data_processed = np.concatenate([armstate,handstate])

            # Numpy array to tensor and changing dictionary keys to LeRobot policy format.
            print(f"qpose:{np.round(qpos_data_processed / np.pi * 180, 1)}")

            #Get the current image.
            current_image = img_array.copy()
            left_image =  current_image[:, :image_w]
            right_image = current_image[:, image_w:]
            img_dic['top'] = get_image_processed(left_image)
            img_dic['wrist'] = get_image_processed(right_image)
            robot_state = qpos_data_processed
            observation["pixels"]= img_dic
            observation["agent_pos"] = robot_state
            observation = preprocess_observation(observation)
            observation['observation.state'] =observation['observation.state'].unsqueeze(0)
            observation = {key: observation[key].to(device, non_blocking=True) for key in observation}
            with torch.inference_mode():
                action = policy.select_action(observation)

            # Convert to CPU / numpy.
            action = action.squeeze(0).to("cpu").numpy()
            print(f"qpose:{np.round(action / np.pi * 180, 1)}")
            if is_single_hand:
                if use_left_hand:
                    left_arm_action = action[:7]
                    left_hand_action = action[-7:]
                    q_pose = np.concatenate([left_arm_action,right_arm_action],axis=0)
                    q_pose[3] = q_pose[3] - 0.12
                else:
                    right_arm_action = action[:7]
                    right_hand_action = action[-7:]
                    q_pose = np.concatenate([left_arm_action,right_arm_action],axis=0)
                    q_pose[3+7] = q_pose[3+7] - 0.12
                
            else:
                arm_action = action[:14]
                left_hand_action = action[14:14+7]
                right_hand_action = action[-(14+7):]
                q_pose = arm_action
                q_pose[3] = q_pose[3] - 0.12
                q_pose[3+7] = q_pose[3+7] - 0.12
            g1_arm.ctrl_dual_arm(q_pose, q_tau_ff)
            tirhand.ctrl(left_hand_action,right_hand_action)

            next_time += period
            sleep_time = next_time - time.time()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                print("Warning: execution time exceeded the desired period")


def main(
    pretrained_policy_path: Path | None = None,
    hydra_cfg_path: str | None = None,
    out_dir: str | None = None,
    config_overrides: list[str] | None = None,
):
    assert (pretrained_policy_path is None) ^ (hydra_cfg_path is None)
    if pretrained_policy_path is not None:
        hydra_cfg = init_hydra_config(str(pretrained_policy_path / "config.yaml"), config_overrides)
    else:
        hydra_cfg = init_hydra_config(hydra_cfg_path, config_overrides)

    if out_dir is None:
        out_dir = f"outputs/eval/{dt.now().strftime('%Y-%m-%d/%H-%M-%S')}_{hydra_cfg.env.name}_{hydra_cfg.policy.name}"

    # Check device is available
    device = get_safe_torch_device(hydra_cfg.device, log=True)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    set_global_seed(hydra_cfg.seed)

    log_output_dir(out_dir)

    logging.info("Making environment.")
    env = make_env(hydra_cfg)

    logging.info("Making policy.")
    if hydra_cfg_path is None:
        policy = make_policy(hydra_cfg=hydra_cfg, pretrained_policy_name_or_path=str(pretrained_policy_path))
    else:
        # Note: We need the dataset stats to pass to the policy's normalization modules.
        policy = make_policy(hydra_cfg=hydra_cfg, dataset_stats=make_dataset(hydra_cfg).stats)

    assert isinstance(policy, nn.Module)
    policy.eval()

    with torch.no_grad(), torch.autocast(device_type=device.type) if hydra_cfg.use_amp else nullcontext():
        info = eval_policy(
            policy
        )
    print(info["aggregated"])

    # Save info
    with open(Path(out_dir) / "eval_info.json", "w") as f:
        json.dump(info, f, indent=2)

    env.close()

    logging.info("End of eval")


if __name__ == "__main__":
    init_logging()

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(
        "-p",
        "--pretrained-policy-name-or-path",
        help=(
            "Either the repo ID of a model hosted on the Hub or a path to a directory containing weights "
            "saved using `Policy.save_pretrained`. If not provided, the policy is initialized from scratch "
            "(useful for debugging). This argument is mutually exclusive with `--config`."
        ),
       default='/home/unitree/Videos/lw/21-53-27_real_world_act_default/checkpoints/100000/pretrained_model'
    )
    group.add_argument(
        "--config",
        help=(
            "Path to a yaml config you want to use for initializing a policy from scratch (useful for "
            "debugging). This argument is mutually exclusive with `--pretrained-policy-name-or-path` (`-p`)."
        ),
    )
    parser.add_argument("--revision", help="Optionally provide the Hugging Face Hub revision ID.")
    parser.add_argument(
        "--out-dir",
        help=(
            "Where to save the evaluation outputs. If not provided, outputs are saved in "
            "outputs/eval/{timestamp}_{env_name}_{policy_name}"
        ),
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Any key=value arguments to override config values (use dots for.nested=overrides)",
    )
    args = parser.parse_args()

    if args.pretrained_policy_name_or_path is None:
        main(hydra_cfg_path=args.config, out_dir=args.out_dir, config_overrides=args.overrides)
    else:
        try:
            pretrained_policy_path = Path(
                snapshot_download(args.pretrained_policy_name_or_path, revision=args.revision)
            )
        except (HFValidationError, RepositoryNotFoundError) as e:
            if isinstance(e, HFValidationError):
                error_message = (
                    "The provided pretrained_policy_name_or_path is not a valid Hugging Face Hub repo ID."
                )
            else:
                error_message = (
                    "The provided pretrained_policy_name_or_path was not found on the Hugging Face Hub."
                )

            logging.warning(f"{error_message} Treating it as a local directory.")
            pretrained_policy_path = Path(args.pretrained_policy_name_or_path)
        if not pretrained_policy_path.is_dir() or not pretrained_policy_path.exists():
            raise ValueError(
                "The provided pretrained_policy_name_or_path is not a valid/existing Hugging Face Hub "
                "repo ID, nor is it an existing local directory."
            )

        main(
            pretrained_policy_path=pretrained_policy_path,
            out_dir=None,
            config_overrides=None,
        )
