import os
import torch
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from utils.data import show_samples
from torch.utils.data import DataLoader


def train_text_conditional_diffusion(
    model,
    dataloader: DataLoader,
    checkpoint_dir: str,
    timesteps: int = 1000,
    num_epochs: int = 50,
    lr: float = 1e-4,
    cfg_dropout: float = 0.1,
    guidance_scale: float = 7.5,
    sample_interval: int = 5,
    device: str = "cuda",
    use_amp: bool = True,
    gradient_accumulation_steps: int = 1,
    resume_from_checkpoint: str | None = None,  # None or 'latest'
):
    """
    Training loop for Text-Conditional with CFG.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    eval_dir = os.path.join(checkpoint_dir, "evaluation")
    os.makedirs(eval_dir, exist_ok=True)

    start_epoch = 0
    training_losses = []
    metrics_history = []
    global_step = 0

    optimizer = optim.AdamW(
        model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=1e-4
    )

    # Cosine annealing with warmup
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=lr * 0.01
    )

    scaler = torch.GradScaler("cuda") if use_amp and device == "cuda" else None

    model.to(device)

    if resume_from_checkpoint:
        if resume_from_checkpoint == "latest":
            save_dir = checkpoint_dir
            checkpoints = [
                f
                for f in os.listdir(save_dir)
                if f.startswith("model_weight_") and f.endswith(".pth")
            ]
            print(checkpoints)
            if checkpoints:
                checkpoints = [
                    f
                    for f in checkpoints
                    if f.startswith(f"model_weight_timesteps_{timesteps}_")
                ]

                epochs = [
                    int(f.split("epoch_")[1].split(".pth")[0]) for f in checkpoints
                ]
                print(epochs)
                latest_epoch = max(epochs)
                resume_from_checkpoint = os.path.join(
                    save_dir,
                    f"model_weight_timesteps_{timesteps}_epoch_{latest_epoch}.pth",
                )
            else:
                print("⚠️  No checkpoints found. Starting from scratch.")
                resume_from_checkpoint = None

        if resume_from_checkpoint and os.path.exists(resume_from_checkpoint):
            print(f"\n📂 Loading checkpoint: {resume_from_checkpoint}")
            checkpoint = torch.load(
                resume_from_checkpoint, map_location=device, weights_only=False
            )
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            if "scheduler_state" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler_state"])

            start_epoch = checkpoint["epoch"]
            training_losses = checkpoint.get("training_losses", [])
            metrics_history = checkpoint.get("metrics_history", [])

            cfg_dropout = checkpoint.get("cfg_dropout", cfg_dropout)
            guidance_scale = checkpoint.get("guidance_scale", guidance_scale)

            print(f"✅ Resumed from epoch {start_epoch}")
            print(f"📊 Training history: {len(training_losses)} epochs")
            print(f"🔬 Metrics history: {len(metrics_history)} evaluations")
        else:
            print(f"⚠️  Checkpoint not found: {resume_from_checkpoint}")
            print("Starting training from scratch.")

    model.train()

    print(
        f"\n🚀 Starting OPTIMIZED training with timesteps {timesteps} for epochs {start_epoch + 1}-{num_epochs}..."
    )
    print(f"📊 CFG Dropout: {cfg_dropout}, Guidance Scale: {guidance_scale}")
    print(
        f"⚡ Mixed Precision: {use_amp}, Gradient Accumulation: {gradient_accumulation_steps}"
    )
    print(f"💾 Checkpoints: {checkpoint_dir}\n")

    for epoch in range(start_epoch, num_epochs):
        epoch_losses = []
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")

        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(device)
            labels = labels.to(device)

            with torch.autocast(device_type="cuda", enabled=use_amp):
                loss, _, _, _ = model(images, labels, cfg_dropout=cfg_dropout)
                loss = loss / gradient_accumulation_steps

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                optimizer.zero_grad()

            epoch_losses.append(loss.item() * gradient_accumulation_steps)

            pbar.set_postfix(
                {
                    "loss": f"{loss.item() * gradient_accumulation_steps:.4f}",
                    "lr": f"{optimizer.param_groups[0]['lr']:.6f}",
                }
            )

            global_step += 1

        pbar.close()

        avg_epoch_loss = np.mean(epoch_losses)
        training_losses.append(avg_epoch_loss)

        print(f"\n📈 Epoch {epoch + 1} Average Loss: {avg_epoch_loss:.4f}")

        scheduler.step()

        # Generate samples
        if (epoch + 1) % sample_interval == 0:
            print("\n🎨 Generating samples...")
            model.eval()

            with torch.no_grad():
                sample_labels = torch.arange(0, 10, device=device)
                with torch.autocast(device_type="cuda", enabled=use_amp):
                    samples = model.sample(
                        labels=sample_labels,
                        shape=(10, 1, 32, 32),
                        device=device,
                        guidance_scale=guidance_scale,
                    )

                sample_path = os.path.join(eval_dir, f"samples_epoch_{epoch + 1}.png")
                show_samples(
                    samples,
                    epoch + 1,
                    labels=sample_labels,
                    title_prefix=f"CFG={guidance_scale}",
                    save_path=sample_path,
                )

                save_dir = checkpoint_dir
                os.makedirs(save_dir, exist_ok=True)
                # Save checkpoint
                ckpt_path = os.path.join(
                    save_dir,
                    f"model_weight_timesteps_{timesteps}_epoch_{epoch + 1}.pth",
                )
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "scheduler_state": scheduler.state_dict(),
                        "loss": avg_epoch_loss,
                        "training_losses": training_losses,
                        "cfg_dropout": cfg_dropout,
                        "guidance_scale": guidance_scale,
                        "metrics_history": metrics_history,
                    },
                    ckpt_path,
                )
                print(f"✅ Checkpoint saved: {ckpt_path}")

            model.train()

    print("\n🎉 Training completed!\n")

    return model, training_losses, metrics_history
