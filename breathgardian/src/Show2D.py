import numpy as np
import matplotlib.pyplot as plt
import torch
    
def show2D(window, tensor, figure, canvas):
    figure.clear()
    rows, cols = 4, 5
    axes = figure.subplots(rows, cols)

    num_images = 20
    slice_indices = torch.linspace(0, tensor.shape[0] - 1, num_images).int()

    for i, ax in enumerate(axes.flatten()):
        slice_idx = slice_indices[i].item()
        img = tensor[slice_idx].numpy()
        ax.imshow(img, cmap="gray")
        ax.set_title(f"Index {slice_idx}", color='#0D47A1')
        ax.axis("off")

    figure.tight_layout()
    canvas.draw()
