import argparse
import json
import logging
import threading
import time
from contextlib import nullcontext
from copy import deepcopy
from datetime import datetime as dt
from pathlib import Path
from typing import Callable

import einops
import gymnasium as gym
import numpy as np
import torch
from datasets import Dataset, Features, Image, Sequence, Value, concatenate_datasets
from huggingface_hub import snapshot_download
from huggingface_hub.utils._errors import RepositoryNotFoundError
from huggingface_hub.utils._validators import HFValidationError
from PIL import Image as PILImage
from torch import Tensor, nn
from tqdm import trange

from lerobot.common.datasets.factory import make_dataset
from lerobot.common.datasets.utils import hf_transform_to_torch
from lerobot.common.envs.factory import make_env
from lerobot.common.envs.utils import preprocess_observation
from lerobot.common.logger import log_output_dir
from lerobot.common.policies.factory import make_policy
from lerobot.common.policies.policy_protocol import Policy
from lerobot.common.policies.utils import get_device_from_parameters
from lerobot.common.utils.io_utils import write_video
from lerobot.common.utils.utils import get_safe_torch_device, init_hydra_config, init_logging, set_global_seed
import cv2
from unitree_dl_utils.device.camera.realsense import RealSenseCamera
from unitree_dds_wrapper.robots import g1

from unitree_dds_wrapper.publisher import Publisher
from unitree_dds_wrapper.subscription import Subscription
from unitree_dds_wrapper.idl import std_msgs


def get_image_processed(cam, img_size=[640, 480]):
    # realsense return cv2 image, BGR format
    curr_images = []
    color_img, _ = cam.get_frame()
    #cv2.imwrite(f"images/{time.time()}.jpg", color_img)
    # color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
    color_img = cv2.resize(color_img, img_size)         # w, h
    #color_img = resize_img_keep_ratio(color_img, (img_size[0], img_size[1]))
    # color_img = np.transpose(color_img, (2,0,1))        # cv2 hwc --> pytorch chw
    # color_img = color_img.astype(np.float32) / 255.0    # [0, 255] --> [0,1]
    curr_images.append(color_img)
    color_img = np.stack(curr_images, axis=0)
    return color_img

class Z1LowCmdPub(Publisher):
    def __init__(self, has_gripper=True):
        super().__init__(message=std_msgs.msg.dds_.String_, topic="rt/z1/cmd")
        self.has_gripper = has_gripper
        self.nq = 7 if self.has_gripper else 6

        self.q = np.zeros(self.nq)
        self.data = {
            "q": [],
            "qd": [],
            "endPose": [] # rpyxyz
        }

    def pre_communication(self): # 发送数据(write)前将数据填充到msg中, Publisher已声明，会自动调用
        self.data['q'] = self.q.tolist()
        self.msg.data = json.dumps(self.data)


class Z1LowStateSub(Subscription):
    def __init__(self):
        super().__init__(message=std_msgs.msg.dds_.String_, topic="rt/z1/state")

        self.data = {
            "q": [],
            "qd": [],
            "endPose": [] # rpyxyz
        }

    def post_communication(self): # 接受回调处理数据, Subscription已声明，会自动调用
        data = json.loads(self.msg.data)
        self.data['q']  = np.array(data["q"])
        self.data['qd']  = np.array(data["qd"])
        self.data['endPose']  = np.array(data["endPose"])

def eval_policy(
    policy: torch.nn.Module,
) -> dict:

    assert isinstance(policy, nn.Module), "Policy must be a PyTorch nn module."
    device = get_device_from_parameters(policy)

    # Reset the policy and environments.
    policy.reset()
    
    # init camera
    print(f"==> init camea")
    # cam0 = RealSenseCamera(serial_number="213322074261") # primart camera
    # cam1 = RealSenseCamera(serial_number="218622274879") # left camera

    cam0 = RealSenseCamera(serial_number="044122071036") # primart camera
    # cam1 = RealSenseCamera(serial_number="218622278527") # left camera
    cam1 = RealSenseCamera(serial_number="218622278527") # left camera

    # init arm
    print(f"==> init arm")
    z1lowcmd =  Z1LowCmdPub()
    z1lowstate = Z1LowStateSub()

    print(f"==> move arm")
    targetPos = np.array([0.2011, 1.3851, -0.7203, 0.9275, -0.0856, 0.1137, -1], dtype=np.float32) # init pos in data collect, 6 dof
    print(f"move to init pos:{targetPos}")
    z1lowcmd.q = targetPos
    z1lowcmd.write()
    time.sleep(2)
    print(f"curr pos:{z1lowstate.data['q']}")

    i=0
    while i in range(100000):
        observation={}
        img_dic=dict()
        # Numpy array to tensor and changing dictionary keys to LeRobot policy format.
        imge0 = get_image_processed(cam0)
        imge1 = get_image_processed(cam1)

        img_dic['top'] = imge0
        img_dic['wrist'] = imge1 

        robot_state = z1lowstate.data['q'].copy()
        observation["pixels"]= img_dic # img_dic
        observation["agent_pos"] = robot_state
        observation = preprocess_observation(observation)
        observation['observation.state'] = observation['observation.state'].unsqueeze(0)

        observation = {key: observation[key].to(device, non_blocking=True) for key in observation}
        with torch.inference_mode():
            action = policy.select_action(observation)

        # Convert to CPU / numpy.
        action = action.squeeze(0).to("cpu").numpy()
        # print("action", action)

        action[-1] = -action[-1] #- 0.027
        z1lowcmd.q = action
        z1lowcmd.write()            
        time.sleep(1/100.0)

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
       default='/home/unitree/gh/diffusion_policy/data/9-23-lerobot/500000/pretrained_model'
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
