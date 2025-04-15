"""
Script Json to Lerobot.

# --raw-dir     Corresponds to the directory of your JSON dataset
# --repo-id     Your unique repo ID on Hugging Face Hub
# --task        The specific task for the dataset (e.g., "pour coffee")
# --push_to_hub Whether or not to upload the dataset to Hugging Face Hub (true or false)
# --robot_type  The type of the robot used in the dataset (e.g., Unitree_G1_Dex3, Unitree_Z1_Dual, Unitree_G1_Dex3)

python unitree_lerobot/utils/convert_unitree_json_to_lerobot.py \
    --repo-id your_name/g1_grabcube_double_hand \
    --output_dir "$HOME/datasets/g1_grabcube_double_hand" 
"""
import os
import tyro
import h5py
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


def lerobot_to_h5(repo_id: str, 
                  output_dir: str,
                  root: str = None,) -> None:
    """Convert lerobot data to HDF5 format."""

    dataset = LeRobotDataset(repo_id=repo_id, root=root,)
    
    for episode_index in tqdm(range(dataset.num_episodes), desc="Episodes", position=0, dynamic_ncols=True):

        from_idx = dataset.episode_data_index["from"][episode_index].item()
        to_idx = dataset.episode_data_index["to"][episode_index].item()

        episode = defaultdict(list)
        cameras = defaultdict(list)

        for step_idx in tqdm(range(from_idx, to_idx), desc=f"Episode {episode_index}", position=1, leave=False, dynamic_ncols=True):
            step = dataset[step_idx]

            image_dict = {
                key.split(".")[2]: np.transpose((value.numpy() * 255).astype(np.uint8), (1, 2, 0))
                for key, value in step.items()
                if key.startswith("observation.image") and len(key.split(".")) >= 3
            }
            for key, value in image_dict.items():
                cameras[key].append(value)

            cam_height, cam_width = next(iter(image_dict.values())).shape[:2]
            episode["state"].append(step["observation.state"])
            episode["action"].append(step["action"])

        episode["cameras"] = cameras
        episode["task"] = step["task"]
        episode_length = to_idx - from_idx
        data_cfg = {
                    'camera_names': list(image_dict.keys()),
                    'cam_height': cam_height,
                    'cam_width': cam_width,
                    'state_dim': np.squeeze(step["observation.state"].numpy().shape),
                    'action_dim': np.squeeze(step["action"].numpy().shape),
                }

        # Prepare data dictionary
        data_dict = {
            '/observations/qpos': [episode["state"]],
            '/observations/qvel': [np.zeros_like(episode["state"])],
            '/action': [episode["action"]],
            **{f'/observations/images/{k}': [v] for k, v in episode["cameras"].items()}
        }

        # Create output directory if needed
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate output path
        h5_path = os.path.join(output_dir, f'episode_{episode_index}.hdf5')

        # Write to HDF5 with compression
        with h5py.File(h5_path, 'w', rdcc_nbytes=1024**2*2, libver='latest') as root:
            # Set attributes
            root.attrs['sim'] = False

            # Create datasets
            obs = root.create_group('observations')
            image = obs.create_group('images')
            
            # Camera images
            for cam_name, images in episode["cameras"].items():
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
            root.create_dataset("language_raw", data=episode["task"])
            substep_reasonings[:] = [episode["task"]] * episode_length
            # Copy all prepared data
            for name, array in data_dict.items():
                root[name][...] = array


if __name__ == "__main__":
    tyro.cli(lerobot_to_h5)

