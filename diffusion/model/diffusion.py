from .unet import UNet
import torch.nn as nn 
from .noise_schedular import NoiseScheduler
import torch 
import torch.nn.functional as F

class CrossAttentionDiffusionModel(nn.Module):
    """
    Complete text-conditional diffusion model with Classifier-Free Guidance (CFG).
    Drop-in replacement for your existing DiffusionModel.
    """
    def __init__(self, in_channels=1, out_channels=1, time_emb_dim=128,
                 text_emb_dim=128, base_channels=64, num_timesteps=1000,
                 num_classes=10, device='cpu'):
        super().__init__()

        self.scheduler = NoiseScheduler(num_timesteps=num_timesteps, device=device)
        self.denoiser = UNet(in_channels, out_channels, time_emb_dim,
                                      text_emb_dim, base_channels, num_classes=num_classes, device=device).to(device) # Ensure denoiser is on device
        self.num_timesteps = num_timesteps
        self.device = device
        self.num_classes = num_classes
        self.null_token = num_classes # num_classes as null token

    def forward(self, x0, labels, cfg_dropout=0.1):
        """
        Training forward pass with Classifier-Free Guidance dropout.
        """
        batch_size = x0.size(0)

        # Sample random timestep
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=x0.device)

        # Add noise
        xt, noise = self.scheduler.add_noise(x0, t)

        # Randomly drop conditioning for CFG training
        mask = torch.rand(batch_size, device=x0.device) < cfg_dropout
        conditioned_labels = torch.where(
            mask,torch.full_like(labels, self.null_token).to(x0.device), 
            labels.to(x0.device),
        )
        pred_noise = self.denoiser(xt, t, conditioned_labels)
        loss = F.mse_loss(pred_noise, noise)

        return loss, pred_noise, noise, xt

    @torch.no_grad()
    def sample(self, labels, shape, device, guidance_scale=7.5):
        """
        Generate images with CFG using DDPM 
        """
        print(f"Sampling on {device}")
        if labels.max() >= self.num_classes:
            raise ValueError(f"Label indices must be in [0, {self.num_classes-1}]")

        # Ensure model is on the correct device
        self.device = device
        self = self.to(device)
        
        # Start from pure noise
        img = torch.randn(shape, device=device)
        batch_size = shape[0]

        # Prepare labels for CFG
        cond_labels = labels.to(device)
        uncond_labels = torch.full((batch_size,), self.null_token, device=device, dtype=torch.long)

        timesteps = list(range(self.num_timesteps - 1, -1, -1))


        # Denoising loop
        for t in timesteps:
            t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)

            # Predict noise with CFG
            if guidance_scale == 1.0:
                # No guidance - just conditional
                pred_noise = self.denoiser(img, t_tensor, cond_labels)
            else:
                # Classifier-free guidance
                pred_noise_uncond = self.denoiser(img, t_tensor, uncond_labels)
                pred_noise_cond = self.denoiser(img, t_tensor, cond_labels)
                pred_noise = pred_noise_uncond + guidance_scale * (pred_noise_cond - pred_noise_uncond)

            img = self.scheduler.denoise_step_ddpm(img, pred_noise, t_tensor)

        return img

    @torch.no_grad()
    def sample_progressive(self, labels, shape, device, guidance_scale=7.5, save_every=10):
        """
        Generate images and save intermediate steps for visualization.
        """
        img = torch.randn(shape, device=device)
        batch_size = shape[0]

        cond_labels = labels.to(device)
        uncond_labels = torch.full((batch_size,), self.null_token, device=device, dtype=torch.long)

        timesteps = list(range(self.num_timesteps - 1, -1, -1))
        intermediates = []

        for i, t in enumerate(timesteps):
            t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)

            # CFG
            if guidance_scale == 1.0:
                pred_noise = self.denoiser(img, t_tensor, cond_labels)
            else:
                pred_noise_uncond = self.denoiser(img, t_tensor, uncond_labels)
                pred_noise_cond = self.denoiser(img, t_tensor, cond_labels)
                pred_noise = pred_noise_uncond + guidance_scale * (pred_noise_cond - pred_noise_uncond)

            # Denoise
            img = self.scheduler.denoise_step_ddpm(img, pred_noise, t_tensor)

            # Save intermediate
            if i % save_every == 0 or i == len(timesteps) - 1:
                intermediates.append(img.clone())
                print(f"  Step {i+1}/{len(timesteps)}: saved intermediate")

        return img, intermediates