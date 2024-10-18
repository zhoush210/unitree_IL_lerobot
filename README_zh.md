**其他语言版本: [English](README.md).**

# 目录说明

| 目录             | 说明                                                                                                                |
| ---------------- | ------------------------------------------------------------------------------------------------------------------- |
| lerobot          | lerobot 仓库代码，已修改可用于 G1 数据转换与训练；其对应的 commit 版本号为 c712d68f6a4fcb282e49185b4af46b0cee6fa5ed |
| unitree_dl_utils | unitree 机器人控制相关代码以及数据处理工具                                                                          |

# 环境安装

## LeRobot 环境安装

本项的目的是使用[LeRobot](https://github.com/huggingface/lerobot)开源框架训练并测试基于 Unitree 机器人采集的数据。所以首先需要安装 LeRobot 相关依赖。安装步骤如下，也可以参考[LeRobot](https://github.com/huggingface/lerobot)官方进行安装:

下载源码

```
cd $HOME
git clone https://github.com/unitreerobotics/unitree_il_lerobot.git
```

创建一个 python 为 3.10 的虚拟环境并激活

```
conda create -y -n lerobot python=3.10
conda activate lerobot
```

安装 LeRobot

```
cd unitree_il_lerobot/lerobot
pip install -e .
```

**注意事项:** 根据您的平台，如果在此步骤中遇到任何构建错误，您可能需要安装 `cmake` 和 `build-essential` 来构建我们的一些依赖项。在 Linux 上，可以使用以下命令安装：`sudo apt-get install cmake build-essential`。

## 机器人控制相关环境安装[可选，真机验证时需要安装]

针对 Unitree 机器人控制需要安装一些依赖,安装步骤如下:

```
git clone https://github.com/unitreerobotics/unitree_dds_wrapper.git
cd unitree_dds_wrapper/python
pip install -e .
```

## 数据下载

如果你想使用我们提供的 Unitree G1 采集的双臂操作的数据集可以访问 [UnitreeG1_DualArmGrasping](https://huggingface.co/datasets/unitreerobotics/UnitreeG1_DualArmGrasping) 。
如需下载可参考下面命令进行下载:

```
cd $HOME
mkdir lerobot_datasets && cd lerobot_datasets
git clone https://huggingface.co/datasets/unitreerobotics/UnitreeG1_DualArmGrasping
```

下载后的数据的存储目录结构如下：

    lerobot_datasets
    └── UnitreeG1_DualArmGrasping
        ├── meta_data
        ├── README.md
        ├── train
        └── videos

# 训练(主要是读取本地数据进行训练)

## 添加本地数据路径

可以使用下面命令设置数据存放路径：

```
export DATA_DIR="$HOME/lerobot_datasets/"
```

## 修改配置文件[可选]

**温馨提示：** 使用我们提供的相关配置文件,可以省略下面的修改

- 修改 `lerobot/lerobot/configs/policy` 中策略的配置
  - `dataset_repo_id` 设置为 `UnitreeG1_DualArmGrasping`(与存储的目录结构有关系)
  - 注意点：
    - 如果使用 ACT 格式的数据集训练 DP，可以在 `lerobot/lerobot/configs/policy` 中添加 `diffusion_aloha.yaml` 配置文件；[参考](https://github.com/huggingface/lerobot/pull/149)
    - 如果生成 DP 的数据集，则需要注意 `override_dataset_stats` 中的 min 和 max
- 修改 `lerobot/lerobot/configs/env` 中环境的策略
  - 主要修改环境配置中的 `state_dim` 和 `action_dim`，修改为自己对应的维度即可

## 运行训练

```
cd unitree_il_lerobot/lerobot
```

- 训练 Diffusion Policy

```
python lerobot/scripts/train.py    policy=diffusion_unitree_real_g1    env=unitree_real_g1     dataset_repo_id=UnitreeG1_DualArmGrasping
```

- 训练 ACT

```
python lerobot/scripts/train.py    policy=act_unitree_real_g1    env=unitree_real_g1     dataset_repo_id=UnitreeG1_DualArmGrasping
```

# 真机测试

在 `lerobot/lerobot/scripts` 中添加 `eval_g1.py`，并运行。

```
python lerobot/lerobot/scripts/eval_g1.py --pretrained-policy-name-or-path "$HOME/unitree_imitation/lerobot/outputs/train/2024-10-17/19-45-30_real_world_act_default/checkpoints/100000/pretrained_modell"
```

**注意:** `--pretrained-policy-name-or-path`根据自己训练的权重位置进行修改； 在`eval_g1.py`中的`eval_policy`函数中有`is_single_hand`变量用于控制是否使用单手或者双手的选项，为`True`是使用单手；`use_left_hand`变量是在使用单手情况下区分使用左手或者右手的，为`True`是使用左手。

**特别提醒:** 如果修改了 LeRobot 的代码，最好是再次进入 `lerobot` 目录中执行`pip install -e .`。

如果使用 Unitree 机器人采集自己的数据并训练可参考下面步骤进行采集和转换。

# 数据采集与转换

## 数据采集

开源的遥操作项目[avp_teleoperate](https://github.com/unitreerobotics/avp_teleoperate)可以使用 Unitree G1 人形机器人进行数据采集，具体可参考[avp_teleoperate](https://github.com/unitreerobotics/avp_teleoperate)项目。

## 数据转换

使用[avp_teleoperate](https://github.com/unitreerobotics/avp_teleoperate)采集的数据是采用 JSON 格式进行存储 ，其结构如下。需要把 JSON 的格式转 lerobot 所需格式，请按照以下步骤进行转换:

以下转换步骤以此数据的存储地址和格式为例。假如采集的数据存放在`$HOME/datasets/`目录中的`g1_grabcube_double_hand` 目录中，格式如下:

    g1_grabcube_double_hand/        #任务名称
    │
    ├── episode_0001                #第一条轨迹
    │    ├──audios/                 #声音信息
    │    ├──colors/                 #图像信息
    │    ├──depths/                 #深度图像信息
    │    └──data.json               #状态以及动作信息
    ├── episode_0002
    ├── episode_...
    ├── episode_xxx

### 数据命名排序

生成 lerobot 的数据集时，最好保证数据的`episode_0`命名是从 0 开始且是连续的，可利用 unitree_utils/sort_and_rename_folders 工具对数据进行排序处理

```
python unitree_utils/sort_and_rename_folders.py --data_dir 'xxx/data/task'
```

- 使用案例:

```
python unitree_utils/sort_and_rename_folders.py --data_dir $HOME/datasets/g1_grabcube_double_hand
```

### 在 Lerobot 源码中添加转换工具[可选]

**注意:** 如果使用我们项目中的 LeRobot，可以省略以下步骤直接进行转换

- 添加 unitree_json_formats 数据转换工具

在 `lerobot/lerobot/common/datasets/push_dataset_to_hub` 中添加 `unitree_json_formats.py`,里面是读取 JSON 数据内容并转成 lerobot 所需数据的工具。(如果需要过滤数据可修改此文件)

**注意:** 其中的 `get_images_data` 函数用于图像数据的处理，需要根据 ACT 或者 DP 配置文件修改 image_key；默认采取的是 ACT 策略中单相机情况的命名`color_0 --> top`，如果是双相机 `color_0 --> top，color_1 --> wrist`

- 导入 unitree_json_formats

为了可以使用导入 `unitree_json_formats` 进行数据转换，需要修改 `lerobot/lerobot/scripts/push_dataset_to_hub.py`。在 `push_dataset_to_hub.py`文件 中的 `get_from_raw_to_lerobot_format_fn` 函数中添加:

```
    elif raw_format=="unitree_json":
        from lerobot.common.datasets.push_dataset_to_hub.unitree_json_format import from_raw_to_lerobot_format
```

### 进行转换

```
python lerobot/scripts/push_dataset_to_hub.py --raw-dir data/ --raw-format unitree_json  --push-to-hub 0 --repo-id lerbot/task --local-dir xxx/videos
```

使用案例:

```
python lerobot/scripts/push_dataset_to_hub.py --raw-dir $HOME/datasets/ --raw-format unitree_json  --push-to-hub 0 --repo-id UnitreeG1_DualArmGrasping --local-dir  $HOME/lerobot_datasets/UnitreeG1_DualArmGrasping --fps 30
```

转换后的数据会存放在 --local-dir 中。

# 致谢

此代码基于以下开源代码库进行构建。请访问以下链接查看相关的许可证：

1. https://github.com/huggingface/lerobot
2. https://github.com/unitreerobotics/unitree_dds_wrapper
