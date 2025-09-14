import torch
import torch.nn as nn
import torch.nn.functional as F

from .scheduler import NoiseScheduler
from .unet import UNet


class DiffusionModel(nn.Module):
    """
    Diffusion Model with UNet & Scheduler
    """

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        time_emb_dim=128,
        base_channels=64,
        num_timesteps=1000,
        device="cpu",
    ):
        super().__init__()
        self.scheduler = NoiseScheduler(
            num_timesteps=num_timesteps, device=device
        )  # forward diffusion process
        self.denoiser = UNet(
            in_channels, out_channels, time_emb_dim, base_channels, device=device
        )  # backward process - noise prediction
        self.num_timesteps = num_timesteps
        self.device = device

    def forward(self, x0):
        """
        One training forward pass:
        - Sample timestep t
        - Add noise to x0 → xt
        - Predict noise using UNet
        - Compute MSE loss between predicted noise and true noise
        """
        batch_size = x0.size(0)

        # sample random timestep for each image
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=x0.device)
        xt, noise = self.scheduler.add_noise(x0, t)  # add noise
        pred_noise = self.denoiser(xt, t)  # predict noise from noisy image
        loss = F.mse_loss(pred_noise, noise)

        return loss, pred_noise, noise, xt

    @torch.no_grad()
    def sample(self, shape, device):
        """
        Reverse diffusion: start from pure noise and gradually denoise and return a generated sample.
        """
        img = torch.randn(shape, device=device)  # x_T ~ N(0, I)

        for t in reversed(range(self.num_timesteps)):
            t_tensor = torch.full((shape[0],), t, device=device, dtype=torch.long)

            # Predict noise at timestep t
            pred_noise = self.denoiser(img, t_tensor)

            # One reverse step
            img = self.scheduler.denoise_step(img, pred_noise, t_tensor)

        return img

    @torch.no_grad()
    def sample_with_intermediate(self, shape, device, timesteps_to_show=None):
        """
        Reverse diffusion: start from pure noise and gradually denoise.
        timesteps_to_show: list of timesteps to save intermediate images
        """
        if timesteps_to_show is None:
            # default: show images at intervals across all timesteps
            timesteps_to_show = [
                0,
                200,
                400,
                600,
                800,
                999,  # last timestep = num_timesteps-1
            ]

        img = torch.randn(shape, device=device)  # x_T ~ N(0, I)
        snapshots = []

        for t in reversed(range(self.num_timesteps)):
            t_tensor = torch.full((shape[0],), t, device=device, dtype=torch.long)

            pred_noise = self.denoiser(img, t_tensor)
            img = self.scheduler.denoise_step(img, pred_noise, t_tensor)

            # Save snapshot if this timestep is in timesteps_to_show
            if t in timesteps_to_show:
                snapshots.append(img.clone())

        return snapshots, timesteps_to_show
