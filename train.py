import os

import torch
import torch.optim as optim

from .utils import show_samples


def train_diffusion(
    model,
    dataloader,
    num_epochs=50,
    lr=1e-4,
    log_interval=800,
    sample_interval=1,
    checkpoint_dir="./checkpoints",
    device="cuda",
):
    os.makedirs(checkpoint_dir, exist_ok=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.to(device)
    step = 0

    for epoch in range(num_epochs):
        for batch_idx, (images, _) in enumerate(dataloader):
            images = images.to(device)

            loss, _, _, _ = model(images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % log_interval == 0:
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}] Step [{batch_idx}/{len(dataloader)}] "
                    f"Loss: {loss.item():.4f}"
                )

            step += 1

        ckpt_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch + 1}.pt")
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            },
            ckpt_path,
        )
        print(f"Saved checkpoint: {ckpt_path}")
        step = 0

        # Generate & show samples
        if (epoch + 1) % sample_interval == 0:
            model.eval()
            with torch.no_grad():
                samples = model.sample((16, 1, 32, 32), device=device)
                show_samples(samples, epoch + 1)
            model.train()
