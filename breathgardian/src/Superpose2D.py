import numpy as np
import matplotlib.pyplot as plt
import torch

def show2D_overlay_on_canvas(ct_volume, mask_volume, figure, canvas, alpha=0.5):
    """
    Display 2D overlays of CT slices with binary mask on a matplotlib canvas.
    
    Args:
        ct_volume (torch.Tensor): 3D tensor (D, H, W) of CT data
        mask_volume (torch.Tensor): 3D tensor (D, H, W) of binary masks (same shape as CT)
        figure (matplotlib.figure.Figure): The Matplotlib figure object
        canvas (FigureCanvas): The canvas to draw the figure on
        alpha (float): Transparency level of mask overlay (0 = transparent, 1 = opaque)
    """
    assert ct_volume.shape == mask_volume.shape, "CT and mask volumes must have the same shape"

    # Define grid layout
    num_images = 20
    rows, cols = 4, 5
    slice_indices = torch.linspace(0, ct_volume.shape[0] - 1, num_images).int()

    # Clear existing figure
    figure.clf()
    grid = figure.add_gridspec(rows, cols)
    axes = [figure.add_subplot(grid[i // cols, i % cols]) for i in range(rows * cols)]

    # Plot each slice with overlay
    for i, ax in enumerate(axes):
        idx = slice_indices[i].item()
        ct_slice = ct_volume[idx].numpy()
        mask_slice = mask_volume[idx].numpy()

        # Normalize CT slice to [0, 1]
        ct_slice = ct_slice.astype(np.float32)
        ct_slice = (ct_slice - ct_slice.min()) / (ct_slice.max() - ct_slice.min() + 1e-8)

        # Convert grayscale to RGB
        ct_rgb = np.stack([ct_slice] * 3, axis=-1)

        # Create red mask overlay
        overlay = ct_rgb.copy()
        overlay[mask_slice == 1] = [1, 0, 0]  # Red

        # Blend overlay with CT using alpha
        blended = (1 - alpha) * ct_rgb + alpha * overlay

        # Display blended image
        ax.imshow(blended)
        ax.set_title(f"Slice {idx}", color='#0D47A1')
        ax.axis("off")

    # Tidy layout and update canvas
    try:
        figure.tight_layout()
    except Exception as e:
        print(f"tight_layout warning: {e}")

    canvas.draw()
