import matplotlib.pyplot as plt


def show_samples(samples, epoch):
    """Helper function to plot sample images in a grid."""
    samples = (samples.clamp(-1, 1) + 1) / 2  # scale back to [0,1]
    samples = samples.cpu()

    fig, axes = plt.subplots(1, 8, figsize=(15, 2))
    for i, ax in enumerate(axes):
        ax.imshow(samples[i].squeeze().cpu().numpy(), cmap="gray")
        ax.axis("off")
    plt.suptitle(f"Samples at Epoch {epoch}", fontsize=14)
    plt.show()
