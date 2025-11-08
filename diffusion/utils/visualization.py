import os
import matplotlib.pyplot as plt
from diffusion.dataset.data_loader import FASHION_CLASSES

def plot_single_sample(sample, label, save_path=None):
    """
    Plot a single generated image with its label.
    
    Args:
        sample (torch.Tensor): Single image tensor of shape (1, 32, 32)
        label (int): Class label for the image
        save_path (str, optional): Path to save the plot. If None, only displays.
    """
    plt.figure(figsize=(3, 3))
    plt.imshow(sample.squeeze().cpu().numpy(), cmap="gray")
    plt.title(f"Label: {FASHION_CLASSES[label]}")
    plt.axis("off")
    
    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
        print(f"Plot saved to {save_path}")
    
    plt.show()
    plt.close()

def plot_random_samples(samples, labels, num_samples=8, save_path=None):
    """
    Plot multiple random samples with their labels.
    
    Args:
        samples (torch.Tensor): Batch of images of shape (N, 1, 32, 32)
        labels (torch.Tensor): Batch of labels
        num_samples (int): Number of samples to plot (default: 8)
        save_path (str, optional): Path to save the plot. If None, only displays.
    """
    # Ensure we don't try to plot more images than we have
    num_samples = min(num_samples, len(samples))
    
    # Calculate figure size
    fig_width = num_samples * 3
    
    # Create subplot grid
    fig, axes = plt.subplots(1, num_samples, figsize=(fig_width, 3), squeeze=False)
    axes = axes.flatten()
    
    for i, ax in enumerate(axes):
        ax.imshow(samples[i].squeeze().cpu().numpy(), cmap="gray")
        ax.set_title(f"Label: {FASHION_CLASSES[labels[i].item()]}")
        ax.axis("off")
    
    plt.tight_layout()
    
    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
        print(f"Plot saved to {save_path}")
    
    plt.show()
    plt.close()