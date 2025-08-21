# 安装
```bash
# 创建 conda 环境
conda create -y -n unitree_lerobot python=3.10
conda activate unitree_lerobot

# 安装 LeRobot
cd unitree_lerobot/lerobot && pip install -e .

# 安装 unitree_lerobot
cd ../.. && pip install -e .

# 安装 unitree_sdk2_python
cd
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git
cd unitree_sdk2_python  && pip install -e .

# 源码安装ffmpeg并开启libsvtav1编译选项
```bash
# 安装 NASM（x86 架构的汇编器）
sudo apt update
sudo apt install nasm
# 安装 SVT-AV1 编码器
cd
git clone https://gitlab.com/AOMediaCodec/SVT-AV1.git
cd SVT-AV1
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON
make -j$(nproc)
sudo make install
# 把这2行加到你的 ~/.bashrc 或 ~/.zshrc 中，以永久生效
export PKG_CONFIG_PATH="/usr/local/lib/pkgconfig:$PKG_CONFIG_PATH" 
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
# 源码安装ffmpeg并开启libsvtav1编译选项
cd
git clone https://github.com/FFmpeg/FFmpeg.git
cd FFmpeg
./configure --enable-libsvtav1
make -j$(nproc)
sudo make install
```

# 数据采集和转换

使用[avp_teleoperate](https://github.com/unitreerobotics/avp_teleoperate)采集的数据是采用 JSON 格式进行存储。假如采集的数据存放在`$HOME/datasets/task_name` 目录中，格式如下，注意不要漏了`task_name`这一层：
```
datasets/                               # 数据集文件夹
    └── task_name /                     # 任务名称
        ├── episode_0001                # 第一条轨迹
        │    ├──audios/                 # 声音信息
        │    ├──colors/                 # 图像信息
        │    ├──depths/                 # 深度图像信息
        │    └──data.json               # 状态以及动作信息
        ├── episode_0002
        ├── episode_...
        ├── episode_xxx
```

生成 lerobot 的数据集时，最好保证数据的`episode_0`命名是从 0 开始且是连续的，使用下面脚本对数据进行排序处理

## 排序和重命名

```bash
python unitree_lerobot/utils/sort_and_rename_folders.py \
        --data_dir $HOME/datasets/task_name
```

## 数据转换

```bash
# --raw-dir     对应json的数据集目录
# --repo-id     对应自己的repo-id 
# --push_to_hub 是否上传到云端 
# --robot_type  对应的机器人类型 Unitree_Z1_Dual, Unitree_G1_Gripper, Unitree_G1_Dex3, Unitree_G1_Dex3_Right_Arm

python unitree_lerobot/utils/convert_unitree_json_to_lerobot.py  
    --raw-dir $HOME/datasets    
    --repo-id your_name/repo_task_name  
    --robot_type Unitree_G1_Dex3 
    --push_to_hub
```
例如：

```bash
# 双臂
python unitree_lerobot/utils/convert_unitree_json_to_lerobot.py \
    --raw-dir /mnt/805_data \
    --repo-id g1/grab_red_bird \
    --robot_type Unitree_G1_Dex3

# 单独右臂
python unitree_lerobot/utils/convert_unitree_json_to_lerobot.py \
    --raw-dir /mnt/805_data \
    --repo-id g1_right_arm/grab_red_bird \
    --robot_type Unitree_G1_Dex3_Right_Arm
```
- 两者需要指定不同的repo_id，后续训练推理时也要使用对应的repo_id

# 训练

- [请详细阅读官方lerobot训练实例与相关参数](https://github.com/huggingface/lerobot/blob/main/examples/4_train_policy_with_script.md)
- --policy.type = act / diffusion / pi0
- 当你在显存小于 30GB 的 GPU 上训练 Pi0 时，建议使用 LoRA。可以通过添加 `--use_lora=true` 来启用。


```bash
# 双臂
cd unitree_lerobot/lerobot
python lerobot/scripts/train.py \
    --dataset.repo_id=g1/grab_red_bird \
    --policy.type=act 

# 单独右臂
cd unitree_lerobot/lerobot
python lerobot/scripts/train.py \
    --dataset.repo_id=g1_right_arm/grab_red_bird \
    --policy.type=act 
```

## 恢复训练

可以在`pretrained_model/config.json`修改某些参数继续训练，也可以直接在命令后指定某些参数
```bash
python lerobot/scripts/train.py \
    --config_path=checkpoint/pretrained_model/ \
    --resume=true \
    --steps=200000
```

# 真机测试

- 控制机器人进入正常运控模式：`L2+B`->`L2+up`->`R1+X`

```bash
# 双臂
python unitree_lerobot/eval_robot/eval_g1/eval_g1.py  \
    --policy.path=unitree_lerobot/lerobot/outputs/train/2025-08-19/11-13-40_act/checkpoints/200000/pretrained_model \
    --repo_id=g1/grab_red_bir

# 单独右臂
python unitree_lerobot/eval_robot/eval_g1/eval_g1.py  \
    --policy.path=unitree_lerobot/lerobot/outputs/train/2025-08-20/14-48-23_act/checkpoints/120000/pretrained_model \
    --repo_id=g1_right_arm/grab_red_bird
```

# 在数据集上测试
```bash
python unitree_lerobot/eval_robot/eval_g1/eval_g1_dataset.py  \
    --policy.path=unitree_lerobot/lerobot/outputs/train/2025-08-19/11-13-40_act/checkpoints/200000/pretrained_model \
    --repo_id=g1/grab_red_bird
```
- 会生成 `figure.png` ，用于可视化 predict action 和 ground truth 的对比曲线，帮助分析模型性能