import torch 


class NoiseScheduler:
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02, device='cpu'):
        self.num_timesteps = num_timesteps
        self.device = device  
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps).to(device) 

        self.alphas = 1.0 - self.betas 
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=device), self.alphas_cumprod[:-1]])

        # Pre-calculated terms for the noise formula
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod).to(device) 
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod).to(device) 


    def add_noise(self, original_image, t):
        t_long = t.long().to(self.device)
        sqrt_alpha_cumprod_t = self.sqrt_alphas_cumprod.gather(-1, t_long).reshape(-1, 1, 1, 1).to(self.device)
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod.gather(-1, t_long).reshape(-1, 1, 1, 1).to(self.device)
        # Sample random noise
        noise = torch.randn_like(original_image, device=self.device)
        noisy_image = sqrt_alpha_cumprod_t * original_image.to(self.device) + sqrt_one_minus_alpha_cumprod_t * noise

        return noisy_image, noise

    def denoise_step_ddpm(self, noisy_image, predicted_noise, t):
        t = t.long().to(self.device)
        alpha_t = self.alphas.gather(0, t).reshape(-1, 1, 1, 1)
        alpha_cumprod_t = self.alphas_cumprod.gather(0, t).reshape(-1, 1, 1, 1)
        beta_t = self.betas.gather(0, t).reshape(-1, 1, 1, 1)
        model_mean = (1 / torch.sqrt(alpha_t)) * (
            noisy_image - (beta_t / torch.sqrt(1 - alpha_cumprod_t)) * predicted_noise
        )
        if (t > 0).any():
            noise = torch.randn_like(noisy_image, device=self.device)
            model_variance = beta_t
            denoised_image = model_mean + torch.sqrt(model_variance) * noise
        else:
            denoised_image = model_mean

        return denoised_image
