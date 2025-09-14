import torch

from diffusion.diffusion import DiffusionModel

from .utils import show_samples

device = "cuda" if torch.cuda.is_available() else "cpu"

model = DiffusionModel()
# Load checkpoint
ckpt = torch.load("./checkpoints/model_epoch_5.pt", map_location=device)
model.load_state_dict(ckpt["model_state"])
model.to(device)

# Generate new samples
model.eval()
with torch.no_grad():
    samples = model.sample((16, 1, 32, 32), device=device)
    samples = (samples.clamp(-1, 1) + 1) / 2
    show_samples(samples, epoch=5)
