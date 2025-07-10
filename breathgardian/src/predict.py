import os
import torch
import torch.nn.functional as F
from monai.networks.nets import UNETR
import numpy as np
import matplotlib.pyplot as plt
import math


# === Load 3D Data ===
def load_3d_image_from_folder(folder_path, dtype=torch.float32):
    """
    Load 2D .npy slices from a folder and convert them into a 3D PyTorch tensor.
    """
    slice_files = [f for f in os.listdir(folder_path) if f.endswith('.npy')]
    if not slice_files:
        raise ValueError(f"No .npy files found in {folder_path}")
    
    slice_files.sort(key=lambda x: int(x.split('-')[-1].split('.')[0]))  # Sort by slice number
    slices = [np.load(os.path.join(folder_path, f)) for f in slice_files]
    volume = np.stack(slices, axis=0)  # (D, H, W)

    return torch.tensor(volume, dtype=dtype)


# === Preprocessing Utilities ===
def pad_to_divisible(tensor, stride=16):
    """
    Pad tensor to make depth, height, width divisible by stride.
    """
    if tensor.dim() == 4:
        tensor = tensor.unsqueeze(1)  # (B, 1, D, H, W)
    _, _, d, h, w = tensor.shape
    d_pad = (stride - d % stride) % stride
    h_pad = (stride - h % stride) % stride
    w_pad = (stride - w % stride) % stride
    return F.pad(tensor, (0, w_pad, 0, h_pad, 0, d_pad), mode="constant", value=0)


def adjust_depth(tensor, target_depth=96):
    """
    Adjusts depth of a 3D tensor to match target_depth by duplicating or skipping slices.
    """
    current_depth = tensor.shape[0]
    if current_depth == target_depth:
        return tensor
    factor = target_depth / current_depth if current_depth < target_depth else current_depth / target_depth
    indices = torch.floor(
        torch.arange(target_depth) / factor if current_depth < target_depth else torch.arange(target_depth) * factor
    ).long()
    return tensor[indices]


def split_tensor_depth(tensor):
    """
    Splits a tensor along depth axis into three parts.
    """
    D, H, W = tensor.shape
    split_sizes = [D // 3, D // 3, D - 2 * (D // 3)]
    return torch.split(tensor, split_sizes, dim=0)


# === Inference Utilities ===
def get_mask_from_data(data_tensor, model_path, threshold=0.8, device=None):
    """
    Predicts a binary segmentation mask from input using a UNETR model.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if data_tensor.dim() == 3:
        data_tensor = data_tensor.unsqueeze(0)  # (1, D, H, W)

    original_shape = data_tensor.shape

    model = UNETR(
        in_channels=1, out_channels=1, img_size=(96, 256, 256),
        feature_size=8, hidden_size=192, mlp_dim=768, num_heads=3,
        norm_name='instance', res_block=True
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device).eval()

    data_tensor = data_tensor.float().to(device)
    padded_data = pad_to_divisible(data_tensor)

    with torch.no_grad():
        output = model(padded_data)
        output = torch.sigmoid(output)

    output = output.squeeze(1)
    d, h, w = original_shape[1:]
    output = output[:, :d, :h, :w]

    return (output > threshold).to(torch.uint8).cpu()


def predict(data_tensor, threshold=0.8, device=None):
    """
    Predicts segmentation mask using 3 UNETR models on 3 depth splits.
    """
    base_dir = os.path.dirname(__file__)
    model_path_0 = os.path.join(base_dir, "model_params", "UNETR_part0_530.pth") 
    model_path_1 = os.path.join(base_dir, "model_params", "UNETR_part1_817.pth")       
    model_path_2 = os.path.join(base_dir, "model_params", "UNETR_part2_150.pth") 

    data_tensor_0, data_tensor_1, data_tensor_2 = split_tensor_depth(data_tensor)

    data_tensor_0 = adjust_depth(data_tensor_0, target_depth=96)
    data_tensor_1 = adjust_depth(data_tensor_1, target_depth=96)
    data_tensor_2 = adjust_depth(data_tensor_2, target_depth=96)

    mask_tensor_0 = get_mask_from_data(data_tensor_0, model_path_0, threshold=threshold, device=device)
    mask_tensor_1 = get_mask_from_data(data_tensor_1, model_path_1, threshold=threshold, device=device)
    mask_tensor_2 = get_mask_from_data(data_tensor_2, model_path_2, threshold=threshold, device=device)

    return torch.cat([
        mask_tensor_0.squeeze(0),
        mask_tensor_1.squeeze(0),
        mask_tensor_2.squeeze(0)
    ], dim=0)


# === Visualization ===
def plot_prediction_overlays(ct_tensor, binary_mask_tensor, num_slices=20):
    """
    Plot CT slices with predicted binary masks overlaid.
    """
    while ct_tensor.dim() > 3:
        ct_tensor = ct_tensor[0]
    while binary_mask_tensor.dim() > 3:
        binary_mask_tensor = binary_mask_tensor[0]

    ct_np = ct_tensor.cpu().numpy()
    mask_np = binary_mask_tensor.cpu().numpy().astype(np.uint8)

    depth = ct_np.shape[0]
    slice_indices = np.linspace(0, depth - 1, num=num_slices, dtype=int)

    cols = min(5, num_slices)
    rows = math.ceil(num_slices / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    for ax_idx, slice_idx in enumerate(slice_indices):
        ax = axes[ax_idx]
        ax.imshow(ct_np[slice_idx], cmap="gray")
        ax.imshow(mask_np[slice_idx], cmap="Blues", alpha=0.4)
        ax.axis("off")

    for ax in axes[len(slice_indices):]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()

