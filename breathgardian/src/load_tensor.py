import os
import numpy as np
import torch

def load_3d_image_from_folder(folder_path, dtype=torch.float32):
    """
    Load 2D .npy slices from a folder and convert them into a 3D PyTorch tensor.

    Args:
        folder_path (str): Path to folder containing 2D slices as .npy files.
        sort_slices (bool): If True, sort slices by filename to preserve order.
        dtype (torch.dtype): Desired tensor data type.

    Returns:
        torch.Tensor: 3D tensor of shape (D, H, W) where D = number of slices.
    """
    # Get all .npy slice files in folder
    slice_files = [f for f in os.listdir(folder_path) if f.endswith('.npy')]

    if not slice_files:
        raise ValueError(f"No .npy files found in {folder_path}")

    
    slice_files.sort(key=lambda x: int(x.split('-')[-1].split('.')[0]))  # Sort by slice number
    
    slices = []
    for file in slice_files:
        slice_path = os.path.join(folder_path, file)
        slice_array = np.load(slice_path)
        slices.append(slice_array)

    # Stack into 3D volume (D, H, W)
    volume = np.stack(slices, axis=0)

    # Convert to PyTorch tensor
    volume_tensor = torch.tensor(volume, dtype=dtype)

    return volume_tensor
