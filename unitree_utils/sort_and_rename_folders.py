import os
import argparse
import uuid

def sort_and_rename_folders(directory):
    # Get the list of folders sorted by name  
    folders = sorted(
        [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]
    )

    temp_mapping = {}

    # First, rename all folders to unique temporary names  
    for folder in folders:
        temp_name = str(uuid.uuid4())
        original_path = os.path.join(directory, folder)
        temp_path = os.path.join(directory, temp_name)
        os.rename(original_path, temp_path)
        temp_mapping[temp_name] = folder

    # Then, rename them to the final target names
    start_number = 0
    for temp_name, original_folder in temp_mapping.items():
        new_folder_name = f'episode_{start_number:04d}'
        temp_path = os.path.join(directory, temp_name)
        new_path = os.path.join(directory, new_folder_name)
        os.rename(temp_path, new_path)
        start_number += 1

    print("The folders have been successfully renamed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sort and rename the folders in the directory")
    parser.add_argument("--data_dir", type=str, required=True, help="The path to the directory containing the folders to be renamed")
    args = parser.parse_args()
    sort_and_rename_folders(args.data_dir)
