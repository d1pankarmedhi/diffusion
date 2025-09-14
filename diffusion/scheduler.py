import torch


class NoiseScheduler:
    """
    Forward Diffusion Process - adding noise to the image
    """

    def __init__(
        self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02, device="cpu"
    ):
        self.num_timesteps = num_timesteps
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps).to(
            device
        )  # creates noise schedule βt

        self.alphas = 1.0 - self.betas  #  $\alpha_t = 1 - \beta_t$
        self.alphas_cumprod = torch.cumprod(
            self.alphas, dim=0
        )  # $\overline{\alpha}$ = $\sum_{s=1}^{t} \alpha_s$ (we keep adding noise step by step)
        self.alphas_cumprod_prev = torch.cat(
            [torch.tensor([1.0], device=device), self.alphas_cumprod[:-1]]
        )

        # Pre-calculated terms for the noise formula
        self.sqrt_alphas_cumprod = torch.sqrt(
            self.alphas_cumprod
        )  # $\sqrt{\overline{\alpha}} x_0 $ (clean image that survives)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(
            1.0 - self.alphas_cumprod
        )  # $\sqrt{1 - \overline{\alpha_t}}ϵ$ (how much noise is mixed in the image)
        self.device = device

    def add_noise(self, original_image, t):
        """
        Adds noise to an image based on a given timestep t.

        Args:
            original_image (torch.Tensor): The clean image (x_0).
            t (torch.Tensor): A tensor of timesteps for each image in the batch.

        Returns:
            tuple: A tuple containing the noisy image (x_t) and the noise that was added.
        """

        #     $x_t = \sqrt{\overline{\alpha}} x_0 + \sqrt{1 - \overline{\alpha_t}}ϵ$

        # Ensure t is a LongTensor for indexing and move to the correct device
        t_long = t.long().to(self.device)

        # Get pre-calculated values for the given timestep t
        sqrt_alpha_cumprod_t = (
            self.sqrt_alphas_cumprod.gather(-1, t_long)
            .reshape(-1, 1, 1, 1)
            .to(self.device)
        )
        sqrt_one_minus_alpha_cumprod_t = (
            self.sqrt_one_minus_alphas_cumprod.gather(-1, t_long)
            .reshape(-1, 1, 1, 1)
            .to(self.device)
        )

        # Sample random noise
        noise = torch.randn_like(original_image, device=self.device)

        # Create the noisy image according to the formula
        noisy_image = (
            sqrt_alpha_cumprod_t * original_image.to(self.device)
            + sqrt_one_minus_alpha_cumprod_t * noise
        )

        return noisy_image, noise

    def denoise_step(self, noisy_image, predicted_noise, t):
        """
        Performs one step of denoising using the reverse process formula.

        Args:
            noisy_image (torch.Tensor): The image at timestep t.
            predicted_noise (torch.Tensor): The noise predicted by the model.
            t (int): The current timestep.

        Returns:
            torch.Tensor: The denoised image (x_{t-1}).
        """
        t = t.long().to(self.device)

        # Gather scalars for current timestep
        alpha_t = self.alphas.gather(0, t).reshape(-1, 1, 1, 1)
        alpha_cumprod_t = self.alphas_cumprod.gather(0, t).reshape(-1, 1, 1, 1)
        beta_t = self.betas.gather(0, t).reshape(-1, 1, 1, 1)

        # Compute mean of reverse distribution
        model_mean = (1 / torch.sqrt(alpha_t)) * (
            noisy_image - (beta_t / torch.sqrt(1 - alpha_cumprod_t)) * predicted_noise
        )

        # If not the last step, add stochasticity
        if (t > 0).any():
            noise = torch.randn_like(noisy_image, device=self.device)
            model_variance = beta_t
            denoised_image = model_mean + torch.sqrt(model_variance) * noise
        else:
            denoised_image = model_mean

        return denoised_image
