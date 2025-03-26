**Read this in other languages: [中文](./docs/README_zh.md).**

# 0. 📖 Introduction

This repository is used for `lerobot training validation` and `unitree related data conversion`

| Directory          | Description                                                                                                                |
| ------------- | ------------------------------------------------------------------------------------------------------------------- |
| lerobot       | The code in the `lerobot repository` for training;  its corresponding commit version number is `725b446a`.|
| utils         | `unitree data processing tool `   |
| eval_robot    | `unitree real machine inference verification of the model`     |


# 1. 📦 Environment Setup

## 1.1 🦾 LeRobot Environment Setup

The purpose of this project is to use the [LeRobot](https://github.com/huggingface/lerobot) open-source framework to train and test data collected from Unitree robots. Therefore, it is necessary to install the LeRobot-related dependencies first. The installation steps are as follows, and you can also refer to the official [LeRobot](https://github.com/huggingface/lerobot) installation guide:

```bash
# Clone the source code
git clone --recurse-submodules https://github.com/unitreerobotics/unitree_IL_lerobot.git

# If already downloaded:
git submodule update --init --recursive

# Create a conda environment
conda create -y -n unitree_lerobot python=3.10
conda activate unitree_lerobot

# Install LeRobot
cd lerobot && pip install -e .

# Install unitree_lerobot
cd .. && pip install -e .
```

## 1.2 🕹️ unitree_sdk2_python

For `DDS communication` on Unitree robots, some dependencies need to be installed. Follow the installation steps below:

```bash
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git
cd unitree_sdk2_python  && pip install -e .
```

# 2. ⚙️ Data Collection and Conversion

## 2.1 🖼️ Load Datasets

Load the `unitreerobotics/G1_ToastedBread_Dataset` dataset from Hugging Face. If you want to load data from a local source, please change the `root` parameter.

```python
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import tqdm

episode_index = 1

dataset = LeRobotDataset(repo_id="unitreerobotics/G1_ToastedBread_Dataset")

from_idx = dataset.episode_data_index["from"][episode_index].item()
to_idx = dataset.episode_data_index["to"][episode_index].item()

for step_idx in tqdm.tqdm(range(from_idx, to_idx)):
    step = dataset[step_idx]
```

`visualization`

```bash
python lerobot/scripts/visualize_dataset.py \
    --repo-id unitreerobotics/G1_ToastedBread_Dataset \
    --episode-index 0
```

## 2.2 🔨 Data Collection

The open-source teleoperation project [avp_teleoperate](https://github.com/unitreerobotics/avp_teleoperate/tree/g1) can be used to collect data using the Unitree G1 humanoid robot. For more details, please refer to the [avp_teleoperate](https://github.com/unitreerobotics/avp_teleoperate/tree/g1) project.

## 2.3 🛠️ Data Conversion

The data collected using [avp_teleoperate](https://github.com/unitreerobotics/avp_teleoperate/tree/g1) is stored in JSON format.

The following conversion steps use this data storage path and format as an example. Assuming the collected data is stored in the `$HOME/datasets/` directory under the `g1_grabcube_double_hand` directory, the format is as follows

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

### 2.3.1 🔀 Sort and Rename

When generating datasets for LeRobot, it is recommended to ensure that the data naming convention, starting from `episode_0`, is sequential and continuous. You can use the `unitree_utils/sort_and_rename_folders` tool to sort and rename the data accordingly.

```bash
python utils/sort_and_rename_folders.py --data_dir $HOME/datasets/g1_grabcube_double_hand
```

#### 2.3.2 🔄 Conversion

Convert `Unitree JSON` Dataset to `LeRobot` Format

```bash
# --raw-dir     Corresponds to the directory of your JSON dataset
# --repo-id     Your unique repo ID on Hugging Face Hub
# --task        The specific task for the dataset (e.g., "pour coffee")
# --push_to_hub Whether or not to upload the dataset to Hugging Face Hub (true or false)
# --robot_type  The type of the robot used in the dataset (e.g., Unitree_G1_Dex3)

python utils/convert_unitree_json_to_lerobot.py 
    --raw-dir $HOME/datasets/g1_grabcube_double_hand    
    --repo-id your_name/g1_grabcube_double_hand 
    --robot_type Unitree_G1_Dex3    # Unitree_Z1_Dual, Unitree_G1_Gripper, Unitree_G1_Dex3
    --task "pour coffee"
    --push_to_hub true
```


# 3. 🚀 Training

[For training, please refer to the official LeRobot training example and parameters for further guidance.](https://github.com/huggingface/lerobot/blob/main/examples/4_train_policy_with_script.md)


- `Train Act Policy`

```bash
python lerobot/scripts/train.py \
    --dataset.repo_id=unitreerobotics/G1_ToastedBread_Dataset \
    --policy.type=act 
```

- `Train Diffusion Policy`

```bash
python lerobot/scripts/train.py \
  --dataset.repo_id=unitreerobotics/G1_ToastedBread_Dataset \
  --policy.type=diffusion
```

- `Train Pi0 Policy`

```bash
python lerobot/scripts/train.py \
  --dataset.repo_id=unitreerobotics/G1_ToastedBread_Dataset \
  --policy.type=pi0
```

# 4. 🛠️ Real-World Testing

To test your trained model on a real robot, you can use the eval_g1.py script located in the eval_robot/eval_g1 folder. Here’s how to run it:

```bash
cd eval_robot/eval_g1

python eval_g1.py  
--policy.path=outputs/train/2025/16_diffusion/checkpoints/100000/pretrained_model 
--repo_id=unitreerobotics/G1_ToastedBread_Dataset
```
# 5. 🤔 Troubleshooting

# 6. 🙏 Acknowledgement

This code builds upon following open-source code-bases. Please visit the URLs to see the respective LICENSES:

1. https://github.com/huggingface/lerobot
2. https://github.com/unitreerobotics/unitree_dds_wrapper
