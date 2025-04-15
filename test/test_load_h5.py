import h5py
import argparse


def read_hdf5(file_path, print_structure=True, print_data=True):
    """
    Read an HDF5 file and print its structure and data.
    
    Args:
        file_path (str): Path to the HDF5 file
        print_structure (bool): Whether to print the file structure
        print_data (bool): Whether to print the data content
    """
    try:
        with h5py.File(file_path, 'r') as f:
            print(f"Successfully opened file: {file_path}")
            h5_name = []
            # Print file structure
            if print_structure:
                print("\nFile structure:")
                def print_attrs(name, obj):
                    print(f"  {name} (Type: {type(obj)})")
                    h5_name.append(name)
                    if isinstance(obj, h5py.Dataset):
                        print(f"    Shape: {obj.shape}, Dtype: {obj.dtype}")
                    for key, val in obj.attrs.items():
                        print(f"    Attribute: {key} = {val}")
                f.visititems(print_attrs)
            
            # Print data content (only partial data to avoid excessive output)
            if print_data:
                for name in h5_name:
                    dataset = f[name]
                    print(f"\nData from {name}:")
                    
                    if isinstance(dataset, h5py.Dataset):
                        # Get dataset shape and type
                        shape = dataset.shape
                        dtype = dataset.dtype
                        print('------------------------------------------------')
                        print(f"Shape: {shape}, Dtype: {dtype}")
                        # Print memory usage
                        print(f"\nTotal dataset size: {dataset.size * dataset.dtype.itemsize / (1024**2):.2f} MB")

                        # Special handling for 4D image data [n, width, height, channels]
                        if len(shape) == 4 and shape[3] == 3:  # Check for 4D with 3 channels
                            print("Image Dataset [width, height, channels]:")
                            
                            # Print statistics
                            sample = dataset[0]  # Take first image
                            print(f"Sample image shape: {sample.shape}")
                            print(f"Pixel value range: {sample.min()} - {sample.max()}")
                            
                            # Print corner pixels (top-left 4x4 area)
                            print("\nTop-left 4x4 corner of first image (R,G,B channels):")
                            print(sample[:4, :4, :])
                            
                            # Print center pixel values
                            center_y, center_x = sample.shape[0]//2, sample.shape[1]//2
                            print("\nCenter pixel values (R,G,B):")
                            print(sample[center_y, center_x, :])
                            
                        else:
                            # Standard data printing for non-image datasets
                            data = dataset[...]
                            if data.size > 10:
                                print("First 10 elements:", data.flatten()[:10], "...")
                            else:
                                print(data)
                    else:
                        print(f"{name} is not a dataset.")
                        
    except Exception as e:
        print(f"Error reading file: {e}")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Read an HDF5 file")
    parser.add_argument("file_path", type=str, help="Path to the HDF5 file")
    parser.add_argument("--no-structure", action="store_true", help="Skip printing file structure")
    parser.add_argument("--no-data", action="store_true", help="Skip printing data content")
    args = parser.parse_args()

    read_hdf5(
        args.file_path,
        print_structure=not args.no_structure,
        print_data=not args.no_data
    )