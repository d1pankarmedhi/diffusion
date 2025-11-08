from torch.utils.data import DataLoader
import matplotlib.pyplot as plt 
from dataset.data_loader import FASHION_CLASSES
import os 
import torch 


def plot_sample_fashionMNIST(data_loader: DataLoader):
    images, labels = next(iter(data_loader))

    # Plot the first few images
    fig, axes = plt.subplots(1, 8, figsize=(15, 2))
    for i, ax in enumerate(axes):
        # Unnormalize and permute dimensions for plotting
        img = images[i].squeeze().cpu().numpy() * 0.5 + 0.5
        ax.imshow(img, cmap="gray")
        ax.set_title(FASHION_CLASSES[labels[i].item()])
        ax.axis("off")
    plt.show()


def show_samples(samples, epoch, labels=None, title_prefix="Samples", save_path=None):
    """
    Helper function to plot sample images in a grid with optional class labels.
    """
    samples = (samples.clamp(-1, 1) + 1) / 2  # Scale back to [0,1]
    samples = samples.cpu()

    num_samples = min(8, len(samples))  # Show max 8 samples
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 2))

    # Handle single image case
    if num_samples == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.imshow(samples[i].squeeze().cpu().numpy(), cmap="gray")
        ax.axis("off")

        # Add class label as subtitle if provided
        if labels is not None:
            label_idx = labels[i].item() if torch.is_tensor(labels[i]) else labels[i]
            if label_idx < len(FASHION_CLASSES):
                ax.set_title(FASHION_CLASSES[label_idx], fontsize=10)

    plt.suptitle(f"{title_prefix} at Epoch {epoch}", fontsize=14)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"💾 Saved samples to: {save_path}")

    plt.show()
    plt.close(fig) # Close the figure to free up memory
