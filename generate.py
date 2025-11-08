import argparse
import torch
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))
from diffusion.pipelines.inference import generate, load_checkpoint
from diffusion.model.diffusion import CrossAttentionDiffusionModel


def generate_samples():
    parser = argparse.ArgumentParser(
        description="Generate samples from a trained diffusion model."
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=False,
        help="Path to the saved model checkpoint dir.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "samples", "generated_samples.png"),
        help="Path to save the output image (PNG format).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run inference on (e.g., 'cpu', 'cuda').",
    )
    parser.add_argument(
        "--single",
        action="store_true",
        help="Generate a single image instead of multiple samples.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=8,
        help="Number of samples to generate (ignored if --single is used).",
    )
    args = parser.parse_args()

    # --- Manual Model Loading without MLflow ---
    print(f"Loading model from path: {args.checkpoint_dir}")
    device = args.device

    model = CrossAttentionDiffusionModel(device=device)
    model = load_checkpoint(
        model=model,
        ckpt_dir=args.checkpoint_dir,
        device=device,
    )
    model.to(device)  
    generate(
        model=model, 
        device=args.device, 
        save_path=args.output_path,
        num_samples=args.num_samples,
        single_image=args.single
    )


if __name__ == "__main__":
    generate_samples()
