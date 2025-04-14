"""
Script Json to Lerobot.

# --raw-dir     Corresponds to the directory of your JSON dataset
# --repo-id     Your unique repo ID on Hugging Face Hub
# --task        The specific task for the dataset (e.g., "pour coffee")
# --push_to_hub Whether or not to upload the dataset to Hugging Face Hub (true or false)
# --robot_type  The type of the robot used in the dataset (e.g., Unitree_G1_Dex3, Unitree_Z1_Dual, Unitree_G1_Dex3)

python unitree_lerobot/utils/convert_unitree_json_to_lerobot.py \
    --raw-dir $HOME/datasets/g1_grabcube_double_hand \
    --repo-id your_name/g1_grabcube_double_hand \
    --robot_type Unitree_G1_Dex3 \ 
    --task "pour coffee" \
    --push_to_hub
"""
import os
import tqdm
import tyro
import h5py
import numpy as np
from collections import defaultdict
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


def lerobot_to_h5(repo_id: str, 
                  output_dir: str,
                  root: str = None,) -> None:
    """Convert lerobot data to HDF5 format."""

    dataset = LeRobotDataset(repo_id=repo_id, root=root,)
    
    for episode_index in tqdm.tqdm(range(dataset.num_episodes)):

        from_idx = dataset.episode_data_index["from"][episode_index].item()
        to_idx = dataset.episode_data_index["to"][episode_index].item()

        episode = defaultdict(list)
        cameras = defaultdict(list)

        for step_idx in tqdm.tqdm(range(from_idx, to_idx)):
            step = dataset[step_idx]

            image_dict = {
                key.split(".")[2]: np.transpose(value.numpy(), (1, 2, 0))
                for key, value in step.items()
                if key.startswith("observation.image") and len(key.split(".")) >= 3
            }
            for key, value in image_dict.items():
                cameras[key].append(value)
            
            # Read cam_height and cam_width from the first image in image_dict
            if image_dict:
                cam_height, cam_width = next(iter(image_dict.values())).shape[:2]
            else:
                cam_height, cam_width = 0, 0

            episode["state"].append(step["observation.state"])
            episode["action"].append(step["action"])

        episode["cameras"] = cameras
        episode["task"] = step["task"]
        episode["episode_length"] = to_idx - from_idx
        episode["episode_idx"] = episode_index
        episode["data_cfg"] = {
                        'camera_names': list(image_dict.keys()),
                        'cam_height': cam_height,
                        'cam_width': cam_width,
                        'state_dim': np.squeeze(step["observation.state"].numpy().shape),
                        'action_dim': np.squeeze(step["action"].numpy().shape),
                    }

        state = episode["state"]
        action = episode["action"]
        qvel = np.zeros_like(episode["state"])
        cameras = episode["cameras"]
        task = episode["task"]
        episode_length = episode["episode_length"]
        episode_idx = episode["episode_idx"]
        data_cfg = episode["data_cfg"]

        # Prepare data dictionary
        data_dict = {
            '/observations/qpos': [state],
            '/observations/qvel': [qvel],
            '/action': [action],
            **{f'/observations/images/{k}': [v] for k, v in cameras.items()}
        }

        # Create output directory if needed
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate output path
        h5_path = os.path.join(output_dir, f'episode_{episode_idx}.hdf5')

        # Write to HDF5 with compression
        with h5py.File(h5_path, 'w', rdcc_nbytes=1024**2*2, libver='latest') as root:
            # Set attributes
            root.attrs['sim'] = False

            # Create datasets
            obs = root.create_group('observations')
            image = obs.create_group('images')
            
            # Camera images
            for cam_name, images in cameras.items():
                image.create_dataset(
                    cam_name,
                    shape=(episode_length, data_cfg['cam_height'], data_cfg['cam_width'], 3),
                    dtype='uint8',
                    chunks=(1, data_cfg['cam_height'], data_cfg['cam_width'], 3),
                    compression="gzip"
                )
                root[f'/observations/images/{cam_name}'][...] = images
            
            # State and action data
            obs.create_dataset('qpos',(episode_length, data_cfg["state_dim"]), dtype='float32', compression="gzip")
            obs.create_dataset('qvel', (episode_length, data_cfg["state_dim"]), dtype='float32', compression="gzip")
            root.create_dataset('action', (episode_length, data_cfg["action_dim"]), dtype='float32', compression="gzip")
            
            # Metadata
            root.create_dataset('is_edited', (1,), dtype='uint8')
            substep_reasonings = root.create_dataset('substep_reasonings', (episode_length,), dtype=h5py.string_dtype(encoding='utf-8'),compression="gzip")
            root.create_dataset("language_raw", data=task)
            substep_reasonings[:] = [task] * episode_length
            # Copy all prepared data
            for name, array in data_dict.items():
                root[name][...] = array


if __name__ == "__main__":
    tyro.cli(lerobot_to_h5)

