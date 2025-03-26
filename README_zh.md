**其他语言版本: [English](README.md).**

# 目录说明

| 目录          | 说明                                                                                                                |
| ------------- | ------------------------------------------------------------------------------------------------------------------- |
| lerobot       | lerobot 仓库代码，其对应的 commit 版本号为 725b446ad6e15499c4d92acce24b72103d4ce777 |
| utils         | unitree 数据处理工具     |
| eval_robot    | unitree 模型真机推理验证     |

# 环境安装

## LeRobot 环境安装

本项的目的是使用[LeRobot](https://github.com/huggingface/lerobot)开源框架训练并测试基于 Unitree 机器人采集的数据。所以首先需要安装 LeRobot 相关依赖。安装步骤如下，也可以参考[LeRobot](https://github.com/huggingface/lerobot)官方进行安装:


```bash
# 下载源码
git clone https://github.com/unitreerobotics/unitree_IL_lerobot.git

# 创建 conda 环境
conda create -y -n unitree_lerobot python=3.10
conda activate unitree_lerobot

# 安装 LeRobot
cd unitree_IL_lerobot/lerobot
pip install -e .
```

**注意事项:** 根据您的平台，如果在此步骤中遇到任何构建错误，您可能需要安装 `cmake` 和 `build-essential` 来构建我们的一些依赖项。在 Linux 上，可以使用以下命令安装：`sudo apt-get install cmake build-essential`。
## Unitree DDS 环境安装
针对 Unitree 机器人控制需要安装一些依赖,安装步骤如下:
```
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git
cd unitree_sdk2_python  && pip install -e .
```

# 数据采集与转换

## 数据加载
加载huggingface上unitreerobotics/G1_ToastedBread_Dataset数据集(v2.0版本)，如果想从加载本地数据使用 root = '.../...' 
```
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import tqdm

episode_index = 1

dataset = LeRobotDataset(repo_id="unitreerobotics/G1_ToastedBread_Dataset")

from_idx = dataset.episode_data_index["from"][episode_index].item()
to_idx = dataset.episode_data_index["to"][episode_index].item()

for step_idx in tqdm.tqdm(range(from_idx, to_idx)):
    step = dataset[step_idx]
```
可视化 
```bash
python lerobot/scripts/visualize_dataset.py \
    --repo-id unitreerobotics/G1_ToastedBread_Dataset \
    --episode-index 0
```

## 数据采集

开源的遥操作项目[avp_teleoperate](https://github.com/unitreerobotics/avp_teleoperate)可以使用 Unitree G1 人形机器人进行数据采集，具体可参考[avp_teleoperate](https://github.com/unitreerobotics/avp_teleoperate)项目。

## 数据转换

使用[avp_teleoperate](https://github.com/unitreerobotics/avp_teleoperate)采集的数据是采用 JSON 格式进行存储 ，其结构如下。需要把 JSON 的格式转 lerobot 所需格式，请按照以下步骤进行转换:

以下转换步骤以此数据的存储地址和格式为例。假如采集的数据存放在`$HOME/datasets/`目录中的`g1_grabcube_double_hand` 目录中，格式如下:

    g1_grabcube_double_hand/        #任务名称
    ├── episode_0001                #第一条轨迹
    │    ├──audios/                 #声音信息
    │    ├──colors/                 #图像信息
    │    ├──depths/                 #深度图像信息
    │    └──data.json               #状态以及动作信息
    ├── episode_0002
    ├── episode_...
    ├── episode_xxx


### 进行转换
生成 lerobot 的数据集时，最好保证数据的`episode_0`命名是从 0 开始且是连续的，可利用 utils/sort_and_rename_folders 工具对数据进行排序处理
```bash
python utils/sort_and_rename_folders.py --data_dir $HOME/datasets/g1_grabcube_double_hand
```


使用 utils/convert_unitree_json_to_lerobot.py 进行转换

```bash
# --raw-dir     对应json的数据集目录
# --repo-id     对应自己的repo-id 
# --task        对应的任务 
# --push_to_hub 是否上传到云端 
# --robot_type  对应的机器人类型 

python utils/convert_unitree_json_to_lerobot.py 
    --raw-dir $HOME/datasets/g1_grabcube_double_hand    
    --repo-id your/g1_grabcube_double_hand 
    --robot_type Unitree_G1_Dex3    # Unitree_Z1_Dual, Unitree_G1_Gripper, Unitree_G1_Dex3
    --task "pour coffee"
    --push_to_hub true
```


# 训练

## 运行训练

[请详细阅读官方lerobot训练实例与相关参数](https://github.com/huggingface/lerobot/blob/main/examples/4_train_policy_with_script.md)


- 训练 act
```
python lerobot/scripts/train.py \
    --dataset.repo_id=unitreerobotics/G1_ToastedBread_Dataset \
    --policy.type=act 
```

- 训练 Diffusion Policy
```
python lerobot/scripts/train.py \
  --dataset.repo_id=unitreerobotics/G1_ToastedBread_Dataset \
  --policy.type=diffusion
```
- 训练 pi0
```
python lerobot/scripts/train.py \
  --dataset.repo_id=unitreerobotics/G1_ToastedBread_Dataset \
  --policy.type=pi0
```

# 真机测试

在 `lerobot/lerobot/scripts` 中添加 `eval_g1.py`，并运行。

```
python lerobot/scripts/eval.py \
    --policy.path=tests/outputs/act/checkpoints/000004/pretrained_model \
    --policy.device=$(DEVICE) \
```

如果使用 Unitree 机器人采集自己的数据并训练可参考下面步骤进行采集和转换。


# 致谢

此代码基于以下开源代码库进行构建。请访问以下链接查看相关的许可证：

1. https://github.com/huggingface/lerobot
2. https://github.com/unitreerobotics/unitree_dds_wrapper
