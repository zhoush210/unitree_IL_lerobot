**Read this in other languages: [中文](README_zh.md).**

# Directory Description

| Directory     | Description                                                                                                                                                                    |
| ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| lerobot       | The code in the lerobot repository has been modified for G1 data conversion and training; its corresponding commit version number is c712d68f6a4fcb282e49185b4af46b0cee6fa5ed. |
| unitree_utils | The code related to Unitree robot control and data processing tools.                                                                                                           |

## LeRobot Environment Setup

The purpose of this project is to use the [LeRobot](https://github.com/huggingface/lerobot) open-source framework to train and test data collected from Unitree robots. Therefore, it is necessary to install the LeRobot-related dependencies first. The installation steps are as follows, and you can also refer to the official [LeRobot](https://github.com/huggingface/lerobot) installation guide:

Download our source code:

```
git clone https://github.com/unitreerobotics/unitree_il_lerobot.git

```

Create a virtual environment with Python 3.10 and activate it, e.g. with miniconda:

```
conda create -y -n lerobot python=3.10
conda activate lerobot
```

Install LeRobot:

```
cd unitree_il_lerobot/lerobot
pip install -e .
```

**NOTE:** Depending on your platform, If you encounter any build errors during this step you may need to install cmake and build-essential for building some of our dependencies. On linux: sudo apt-get install cmake build-essential

## Robot Control Environment Setup

To control the Unitree robot, some dependencies need to be installed. The installation steps are as follows:

```
git clone https://github.com/unitreerobotics/unitree_dds_wrapper.git
cd unitree_dds_wrapper/python
pip install -e .
```

## Data Download

If you would like to use the dual-arm operation dataset collected with the Unitree G1 that we provide, you can visit [UnitreeG1_DualArmGrasping](https://huggingface.co/datasets/unitreerobotics/UnitreeG1_DualArmGrasping) .
To download it, you can refer to the following command:

```
git clone https://huggingface.co/datasets/unitreerobotics/UnitreeG1_DualArmGrasping
```

# Data Collection and Conversion

## Data Collection

The open-source teleoperation project [avp_teleoperate](https://github.com/unitreerobotics/avp_teleoperate) can be used to collect data using the Unitree G1 humanoid robot. For more details, please refer to the [avp_teleoperate](https://github.com/unitreerobotics/avp_teleoperate) project.

## Data Conversion

The data collected using [avp_teleoperate](https://github.com/unitreerobotics/avp_teleoperate) is stored in JSON format, with the structure as shown below. To convert the JSON format into the format required by lerobot, please follow the steps below:

The following conversion steps use this data's storage location and format as an example. If there is only one `g1_grabcube_double_hand` directory in `/home/unitree/datasets/` and it stores the collected dataset, the format is as follows:

    g1_grabcube_double_hand/        # Task name
    │
    ├── episode_0001                # First trajectory
    │    ├──audios/                 # Audio information
    │    ├──colors/                 # Image information
    │    ├──depths/                 # Depth image information
    │    └──data.json               # State and action information
    ├── episode_0002
    ├── episode_...
    ├── episode_xxx

### Data Naming and Sorting

When generating datasets for LeRobot, it is recommended to ensure that the data naming convention, starting from `episode_0`, is sequential and continuous. You can use the `unitree_utils/sort_and_rename_folders` tool to sort and rename the data accordingly.

```
python unitree_utils/sort_and_rename_folders.py --data_dir 'xxx/data/task'
```

- Example Usage:

```
python unitree_utils/sort_and_rename_folders.py --data_dir /home/unitree/datasets/g1_grabcube_double_hand
```

### Add Conversion Tool in Lerobot Source Code (optional)

**Notes:** If you're using LeRobot in our project, you can skip the following steps and directly perform the conversion.

- Add unitree_json_formats Data Conversion Tool

Add `unitree_json_formats.py` in `lerobot/lerobot/common/datasets/push_dataset_to_hub`. This file contains the tool that reads JSON data and converts it into the format required by LeRobot. (If filtering of the data is needed, you can modify this file.)

**Notes:** The `get_images_data` function is responsible for processing image data. You may need to modify the `image_key` based on the ACT or DP configuration file. By default, it uses the ACT strategy for single camera situations with the naming convention `color_0 --> top`. For dual camera setups, the naming would be `color_0 --> top` and `color_1 --> wrist`.

- Import unitree_json_formats

To enable the use of `unitree_json_formats` for data conversion, you need to modify `lerobot/lerobot/scripts/push_dataset_to_hub.py`. Add the following line in the `get_from_raw_to_lerobot_format_fn` function in `push_dataset_to_hub.py`:

```
    elif raw_format=="unitree_json":
        from lerobot.common.datasets.push_dataset_to_hub.unitree_json_format import from_raw_to_lerobot_format
```

### Perform the Conversion

```
python push_dataset_to_hub.py --raw-dir data/ --raw-format unitree_json  --push-to-hub 0 --repo-id lerbot/task --local-dir xxx/videos
```

- Example Usage:

```
python lerobot/scripts/push_dataset_to_hub.py --raw-dir /home/unitree/datasets/ --raw-format unitree_json  --push-to-hub 0 --repo-id unitree/g1_grabcube_double_hand --local-dir /home/unitree/lerobot_datasets/unitree/g1_grabcube_double_hand --fps 30
```

The converted data will be stored in the directory specified by `--local-dir`.

# Training (Primarily Reading Local Data for Training)

Assume the structure of the directory where the converted data is stored is as follows:

    lerobot_datasets/
    │
    ├── unitree
    │    ├──g1_grabcube_double_hand/
    │       ├── meta_data/
    │       ├── train/
    │       └── videos/

## Add Local Data Path

Use the following command to set the data storage path:

```
export DATA_DIR="XXX"
```

As an example, for the above dataset directory, you can set the `DATA_DIR` as follows:

```
export DATA_DIR="/home/unitree/lerobot_datasets/"
```

## Modify Configuration Files(optional)

**Friendly Reminder:** By using the configuration files we provide, you can skip the modifications below.

- Modify the policy configuration in `lerobot/lerobot/configs/policy`:
  - Set `dataset_repo_id` to `unitree/g1_grabcube_double_hand` (this relates to the storage directory structure).
    - Notes:
      - If training DP using a dataset in ACT format, you can add the `diffusion_aloha.yaml` configuration file in `lerobot/lerobot/configs/policy`; [reference](https://github.com/huggingface/lerobot/pull/149).
      - If generating a dataset for DP, be mindful of `min` and `max` in `override_dataset_stats`.
- Modify the environment policy in `lerobot/lerobot/configs/env`:
  - Mainly adjust `state_dim` and `action_dim` in the environment configuration to match the required dimensions.

## Run Training

```
python lerobot/scripts/train.py \
    policy=act \
    env=aloha \
    dataset_repo_id=xxx
```

### Example Use Cases:

- Training Diffusion Policy:

```
python lerobot/scripts/train.py    policy=diffusion_unitree_real_g1    env=unitree_real_g1     dataset_repo_id=unitree/g1_grabcube_double_hand
```

- Training ACT:

```
python lerobot/scripts/train.py    policy=act_unitree_real_g1    env=unitree_real_g1     dataset_repo_id=unitree/g1_grabcube_double_hand
```

# 4. Real Robot Testing

In `lerobot/lerobot/scripts`, add the `eval_g1.py` script and then run it.

```
python lerobot/lerobot/scripts/eval_g1.py --pretrained-policy-name-or-path "/home/unitree/datasets/21-53-27_real_world_act_default/checkpoints/100000/pretrained_model"
```

**Note:** In the `eval_g1.py` script, the `eval_policy` function contains a `is_single_hand` variable that controls whether to use a single hand or both hands. If `is_single_hand` is set to True, it indicates the use of a single hand. The `use_left_hand` variable is used to distinguish between the left or right hand when using a single hand. If `use_left_hand` is True, it signifies the use of the left hand.

**Special reminder:** If you have modified the LeRobot code, it is recommended to re-enter the `lerobot` directory and run `pip install -e .`.

# Acknowledgement

This code builds upon following open-source code-bases. Please visit the URLs to see the respective LICENSES:

1. https://github.com/huggingface/lerobot
2. https://github.com/unitreerobotics/unitree_dds_wrapper
