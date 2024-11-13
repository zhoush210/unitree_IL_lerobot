**他の言語で読む: [English](README.md), [中文](README_zh.md).**

# ディレクトリの説明

| ディレクトリ     | 説明                                                                                                                |
| ------------- | ------------------------------------------------------------------------------------------------------------------- |
| lerobot       | lerobot リポジトリのコードは G1 データの変換とトレーニングに使用するために修正されています。その対応するコミットバージョン番号は c712d68f6a4fcb282e49185b4af46b0cee6fa5ed です。 |
| unitree_utils | Unitree ロボットの制御およびデータ処理ツールに関連するコード。                                                                          |

# 環境設定

## LeRobot 環境設定

このプロジェクトの目的は、[LeRobot](https://github.com/huggingface/lerobot) オープンソースフレームワークを使用して、Unitree ロボットから収集したデータをトレーニングおよびテストすることです。したがって、最初に LeRobot 関連の依存関係をインストールする必要があります。インストール手順は以下の通りです。また、公式の [LeRobot](https://github.com/huggingface/lerobot) インストールガイドも参照できます。

ソースコードをダウンロード:

```
cd $HOME

git clone git@github.com:unitreerobotics/unitree_IL_lerobot.git

または

git clone https://github.com/unitreerobotics/unitree_IL_lerobot.git

```

Python 3.10 の仮想環境を作成してアクティブにする（例：miniconda を使用）:

```
conda create -y -n lerobot python=3.10
conda activate lerobot
```

LeRobot をインストール:

```
cd unitree_IL_lerobot/lerobot
pip install -e .
```

**注意:** プラットフォームに応じて、この手順でビルドエラーが発生した場合は、いくつかの依存関係をビルドするために `cmake` と `build-essential` をインストールする必要があるかもしれません。Linux では次のコマンドを使用してインストールできます: `sudo apt-get install cmake build-essential`

## ロボット制御環境の設定[オプション、実ロボットテストにはインストールが必要]

Unitree ロボットを制御するためには、いくつかの依存関係をインストールする必要があります。インストール手順は以下の通りです:

```
git clone https://github.com/unitreerobotics/unitree_dds_wrapper.git
cd unitree_dds_wrapper/python
pip install -e .
```

## データのダウンロード

提供されている Unitree G1 で収集されたデュアルアーム操作のデータセットを使用したい場合は、[UnitreeG1_DualArmGrasping](https://huggingface.co/datasets/unitreerobotics/UnitreeG1_DualArmGrasping) を訪問してください。
ダウンロードするには、以下のコマンドを参照してください:

```
cd $HOME
mkdir lerobot_datasets && cd lerobot_datasets
git clone https://huggingface.co/datasets/unitreerobotics/UnitreeG1_DualArmGrasping
```

**注意:** ダウンロードに問題が発生した場合は、git-lfs をインストールする必要があるかもしれません。インストールコマンドは以下の通りです:

```
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs
git lfs install
```

ダウンロードされたデータの保存ディレクトリ構造は以下の通りです:

    lerobot_datasets
    └── UnitreeG1_DualArmGrasping
        ├── meta_data
        ├── README.md
        ├── train
        └── videos

# トレーニング（主にローカルデータを読み取ってトレーニング）

## ローカルデータパスの追加

以下のコマンドを使用してデータ保存パスを設定します:

```
export DATA_DIR="$HOME/lerobot_datasets/"
```

## 設定ファイルの変更[オプション]

**親切なリマインダー:** 提供されている設定ファイルを使用することで、以下の変更をスキップできます。

- `lerobot/lerobot/configs/policy` のポリシー設定を変更:
  - `dataset_repo_id` を `UnitreeG1_DualArmGrasping` に設定（これは保存ディレクトリ構造に関連しています）。
    - 注意点:
      - ACT 形式のデータセットを使用して DP をトレーニングする場合、`lerobot/lerobot/configs/policy` に `diffusion_aloha.yaml` 設定ファイルを追加できます。[参考](https://github.com/huggingface/lerobot/pull/149)
      - DP のデータセットを生成する場合、`override_dataset_stats` の min と max に注意してください。
- `lerobot/lerobot/configs/env` の環境ポリシーを変更:
  - 主に環境設定の `state_dim` と `action_dim` を調整し、必要な次元に合わせます。

## トレーニングの実行

```
cd unitree_IL_lerobot/lerobot
```

- Diffusion Policy のトレーニング:

```
python lerobot/scripts/train.py    policy=diffusion_unitree_real_g1    env=unitree_real_g1     dataset_repo_id=UnitreeG1_DualArmGrasping
```

- ACT のトレーニング:

```
python lerobot/scripts/train.py    policy=act_unitree_real_g1    env=unitree_real_g1     dataset_repo_id=UnitreeG1_DualArmGrasping
```

# 実ロボットテスト

`lerobot/lerobot/scripts` に `eval_g1.py` スクリプトを追加し、次に実行します。

```
python lerobot/lerobot/scripts/eval_g1.py --pretrained-policy-name-or-path "$HOME/unitree_imitation/lerobot/outputs/train/2024-10-17/19-45-30_real_world_act_default/checkpoints/100000/pretrained_modell"
```

**注意:** `--pretrained-policy-name-or-path` はトレーニングされた重みの保存場所に応じて変更してください。`eval_g1.py` スクリプトの `eval_policy` 関数には、片手または両手を使用するかどうかを制御する `is_single_hand` 変数があります。`is_single_hand` が True に設定されている場合は片手を使用します。`use_left_hand` 変数は片手を使用する場合に左手または右手を区別するために使用されます。`use_left_hand` が True の場合は左手を使用します。

**特別なリマインダー:** LeRobot のコードを変更した場合は、再度 `lerobot` ディレクトリに入り、`pip install -e .` を実行することをお勧めします。

Unitree ロボットを使用して独自のデータを収集し、トレーニングする場合は、以下の手順を参照してデータ収集と変換を行ってください。

# データ収集と変換

## データ収集

オープンソースの遠隔操作プロジェクト [avp_teleoperate](https://github.com/unitreerobotics/avp_teleoperate/tree/g1) を使用して、Unitree G1 ヒューマノイドロボットを使用してデータを収集できます。詳細については、[avp_teleoperate](https://github.com/unitreerobotics/avp_teleoperate/tree/g1) プロジェクトを参照してください。

## データ変換

[avp_teleoperate](https://github.com/unitreerobotics/avp_teleoperate/tree/g1) を使用して収集されたデータは JSON 形式で保存され、その構造は以下の通りです。JSON 形式を lerobot が必要とする形式に変換するには、以下の手順に従ってください。

以下の変換手順は、このデータの保存パスと形式を例にしています。収集されたデータが `$HOME/datasets/` ディレクトリの `g1_grabcube_double_hand` ディレクトリに保存されていると仮定します。形式は以下の通りです:

    g1_grabcube_double_hand/        # タスク名
    │
    ├── episode_0001                # 最初の軌跡
    │    ├──audios/                 # 音声情報
    │    ├──colors/                 # 画像情報
    │    ├──depths/                 # 深度画像情報
    │    └──data.json               # 状態およびアクション情報
    ├── episode_0002
    ├── episode_...
    ├── episode_xxx

### データの命名とソート

LeRobot のデータセットを生成する際には、データの命名規則が `episode_0` から始まり、連続していることを確認することをお勧めします。`unitree_utils/sort_and_rename_folders` ツールを使用してデータをソートおよびリネームできます。

```
cd unitree_IL_lerobot
python unitree_utils/sort_and_rename_folders.py --data_dir $HOME/datasets/g1_grabcube_double_hand
```

### Lerobot ソースコードにデータ変換ツールを追加[オプション]

**注意:** プロジェクト内の LeRobot を使用している場合、以下の手順をスキップして直接変換を実行できます。

- unitree_json_formats データ変換ツールを追加

`lerobot/lerobot/common/datasets/push_dataset_to_hub` に `unitree_json_formats.py` を追加します。このファイルには、JSON データを読み取り、LeRobot が必要とする形式に変換するツールが含まれています。（データのフィルタリングが必要な場合は、このファイルを変更できます。）

**注意:** `get_images_data` 関数は画像データの処理を担当します。ACT または DP 設定ファイルに基づいて `image_key` を変更する必要があるかもしれません。デフォルトでは、ACT 戦略の単一カメラ状況での命名規則 `color_0 --> top` を使用します。デュアルカメラ設定の場合、命名は `color_0 --> top` および `color_1 --> wrist` となります。

- unitree_json_formats をインポート

データ変換に `unitree_json_formats` を使用できるようにするために、`lerobot/lerobot/scripts/push_dataset_to_hub.py` を変更する必要があります。`push_dataset_to_hub.py` ファイルの `get_from_raw_to_lerobot_format_fn` 関数に以下の行を追加します:

```
    elif raw_format=="unitree_json":
        from lerobot.common.datasets.push_dataset_to_hub.unitree_json_format import from_raw_to_lerobot_format
```

### 変換の実行

```
cd unitree_IL_lerobot/lerobot
python lerobot/scripts/push_dataset_to_hub.py --raw-dir $HOME/datasets/ --raw-format unitree_json  --push-to-hub 0 --repo-id UnitreeG1_DualArmGrasping --local-dir  $HOME/lerobot_datasets/UnitreeG1_DualArmGrasping --fps 30
```

**注意:** 変換されたデータは `--local-dir` に保存されます。`--repo-id` は必要に応じて記入できます。

変換後、上記のトレーニング手順を参照してトレーニングとテストを行うことができます。

# 謝辞

このコードは以下のオープンソースコードベースに基づいて構築されています。関連するライセンスを確認するには、以下の URL を訪問してください:

1. https://github.com/huggingface/lerobot
2. https://github.com/unitreerobotics/unitree_dds_wrapper
