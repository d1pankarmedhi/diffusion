import os
import time
import torch
from diffusion.model.diffusion import CrossAttentionDiffusionModel
from diffusion.dataset.data_loader import FASHION_CLASSES


def load_checkpoint(
    model: CrossAttentionDiffusionModel,
    ckpt_dir: str,
    timesteps: int = 1000,
    device: str = "cpu",
):
    print(f"Checkpoint dir: {ckpt_dir}")
    print(f"Timesteps: {timesteps} | Device: {device}")
    
    checkpoints = [
        f
        for f in os.listdir(ckpt_dir)
        if f.startswith("model_weight_") and f.endswith(".pth")
    ]
    print(f"Checkpoints found: {checkpoints}")

    if checkpoints:
        checkpoints = [
            f
            for f in checkpoints
            if f.startswith(f"model_weight_timesteps_{timesteps}_")
        ]

        epochs = [int(f.split("epoch_")[1].split(".pth")[0]) for f in checkpoints]
        latest_epoch = max(epochs)
        ckpt_path = os.path.join(
            ckpt_dir,
            f"model_weight_timesteps_{timesteps}_epoch_{latest_epoch}.pth",
        )
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        print("Model loaded!")
        return model.to(device)
    else:
        print("No checkpoints found! Loading Default.")
        return model.to(device)


def generate(
    model: CrossAttentionDiffusionModel,
    save_path: str,
    device: str = "cuda",
    num_samples: int = 8,
    single_image: bool = False
):
    """
    Generate samples from the model.
    """
    from diffusion.utils.visualization import plot_single_sample, plot_random_samples
    
    # Ensure the model and all its submodules are on the correct device
    model = model.to(device)
    for param in model.parameters():
        param.data = param.data.to(device)
    model.eval()

    # Generate either single or multiple samples
    batch_size = 1 if single_image else num_samples
    labels = torch.randint(0, 10, (batch_size,), device=device)

    st = time.time()
    with torch.no_grad():
        samples = model.sample(labels, (batch_size, 1, 32, 32), device=device)
    et = time.time()
    print(f"Time taken: {et - st:.2f} seconds")

    # Use appropriate visualization function
    if single_image:
        plot_single_sample(samples[0], labels[0].item(), save_path)
    else:
        plot_random_samples(samples, labels, num_samples, save_path)
