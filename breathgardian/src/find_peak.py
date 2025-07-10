import numpy as np
import matplotlib.pyplot as plt

def plot_value_occurrences(tensor, bins=100):
    """
    Plots the curve of value occurrences in a 3D tensor.
    
    Parameters:
    tensor (np.ndarray): A 3D NumPy array.
    bins (int): Number of bins for histogram calculation.
    """
    if tensor.ndim != 3:
        raise ValueError("Input tensor must be 3-dimensional")

    # Flatten the tensor into a 1D vector
    flat = tensor.flatten()

    # Compute histogram (counts per bin)
    counts, bin_edges = np.histogram(flat, bins=bins)

    # Plot as a curve
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    plt.plot(bin_centers, counts, marker='o')
    plt.title("Value Occurrence Curve")
    plt.xlabel("Value")
    plt.ylabel("Number of Occurrences")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def find_second_bump_limits(tensor, bins=100, plot=False, threshold=0.5):
    """
    Finds the start and end of the second 'bump' (continuous high-density zone) in the histogram.

    Args:
        tensor (np.ndarray): Input 3D tensor.
        bins (int): Number of histogram bins. Default is 100.
        plot (bool): If True, plot the histogram with detected bumps.
        threshold (float): Fraction of max density to consider as 'high'. Default is 0.5.

    Returns:
        tuple: (start_value, end_value) of the second bump.
    """
    vectorized = tensor.flatten()

    # Compute histogram
    counts, bin_edges = np.histogram(vectorized, bins=bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    max_density = np.max(counts)
    high_density_mask = counts > (threshold * max_density)

    # Find continuous regions of high density
    bumps = []
    start = None
    for i in range(len(high_density_mask)):
        if high_density_mask[i] and start is None:
            start = i
        elif not high_density_mask[i] and start is not None:
            bumps.append((start, i - 1))
            start = None
    if start is not None:
        bumps.append((start, len(high_density_mask) - 1))

    if len(bumps) < 2:
        raise ValueError("Less than two bumps detected.")

    # Get the second bump
    bump_start_idx, bump_end_idx = bumps[1]
    bump_start = bin_centers[bump_start_idx]
    bump_end = bin_centers[bump_end_idx]

    if plot:
        plt.plot(bin_centers, counts)
        for start_idx, end_idx in bumps:
            plt.axvspan(bin_centers[start_idx], bin_centers[end_idx], color='red', alpha=0.3)
        plt.plot([bump_start, bump_end], [0, 0], 'rx', markersize=10)  # start and end
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.title('Histogram with Detected Bumps')
        plt.show()

    return bump_start, bump_end
