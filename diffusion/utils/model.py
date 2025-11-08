import matplotlib.pyplot as plt 
import os 
import torch 
from .data import show_samples


def plot_training_curve(training_losses, save_path="./training_curve.png"):
    """
    Plot and save training loss curve.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(training_losses) + 1), training_losses,
             linewidth=2, marker='o', markersize=4)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Average Loss", fontsize=12)
    plt.title("Text-Conditional Diffusion Training Loss", fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"📊 Training curve saved to: {save_path}")
    plt.show()


def test_specific_generation(model, device="cuda", guidance_scale=7.5, save_dir="./test_samples"):
    """
    Interactive function to generate specific Fashion-MNIST items.
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    print("\n" + "="*60)
    print("🎨 Text-Conditional Fashion-MNIST Generator")
    print("="*60)

    examples = [
        ("4 Sneakers", [7, 7, 7, 7]),
        ("Mixed: Dress, Coat, Bag, Boot", [3, 4, 8, 9]),
        ("Complete Wardrobe", list(range(10))),
    ]

    for description, label_list in examples:
        print(f"\n📝 {description}")
        labels = torch.tensor(label_list, device=device)

        with torch.no_grad():
            samples = model.sample(
                labels=labels,
                shape=(len(label_list), 1, 32, 32),
                device=device,
                guidance_scale=guidance_scale
            )

        save_path = os.path.join(save_dir, f"{description.replace(' ', '_').replace(':', '')}.png")
        show_samples(samples, epoch="Test", labels=labels,
                    title_prefix=description, save_path=save_path)

    print("\n" + "="*60 + "\n")