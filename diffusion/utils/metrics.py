import os
from typing import Any, cast
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
from scipy import linalg
from torchvision.models import inception_v3
from tqdm import tqdm
from dataset.data_loader import FASHION_CLASSES

classes = FASHION_CLASSES.items()


def show_samples(samples, epoch, labels=None, title_prefix="Samples", save_path=None):
    """
    Helper function to plot sample images in a grid with optional class labels.
    """
    samples = (samples.clamp(-1, 1) + 1) / 2  # Scale back to [0,1]
    samples = samples.cpu()

    num_samples = min(16, len(samples))
    grid_size = int(np.ceil(np.sqrt(num_samples)))

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    axes = axes.flatten() if num_samples > 1 else [axes]

    for i in range(len(axes)):
        if i < num_samples:
            # Handle grayscale and RGB
            img = samples[i].squeeze()
            if img.dim() == 3 and img.shape[0] == 1:
                img = img.squeeze(0)
            axes[i].imshow(img.numpy(), cmap="gray" if img.dim() == 2 else None)

            # Add class label as subtitle if provided
            if labels is not None:
                label_idx = (
                    labels[i].item() if torch.is_tensor(labels[i]) else labels[i]
                )
                if label_idx < len(FASHION_CLASSES):
                    axes[i].set_title(FASHION_CLASSES[label_idx], fontsize=8)
        axes[i].axis("off")

    plt.suptitle(f"{title_prefix} at Epoch {epoch}", fontsize=14)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"💾 Saved samples to: {save_path}")

    plt.show()
    plt.close(fig)


class InceptionV3FeatureExtractor(nn.Module):
    """
    Pretrained InceptionV3 for extracting features AND logits
    """

    def __init__(self, device="cuda", use_fp16=False):
        super().__init__()
        self.device = device
        self.use_fp16 = use_fp16

        # Load pretrained Inception v3
        self.inception = inception_v3(pretrained=True, transform_input=False)
        self.inception.eval()
        self.inception = self.inception.to(device)

        if use_fp16:
            self.inception = self.inception.half()

        # Freeze parameters
        for param in self.inception.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def get_features(self, x):
        """
        Extract 2048-dimensional features (for FID/KID).

        Args:
            x: Input images [B, C, H, W] in range [0, 1]
        Returns:
            features: [B, 2048] feature vectors
        """
        # Resize to 299x299
        if x.shape[-1] != 299:
            x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)

        # Convert grayscale to RGB
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        # Normalize to [-1, 1]
        x = 2 * x - 1

        if self.use_fp16:
            x = x.half()

        # Extract features from the pool layer
        x = self.inception.Conv2d_1a_3x3(x)
        x = self.inception.Conv2d_2a_3x3(x)
        x = self.inception.Conv2d_2b_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.inception.Conv2d_3b_1x1(x)
        x = self.inception.Conv2d_4a_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.inception.Mixed_5b(x)
        x = self.inception.Mixed_5c(x)
        x = self.inception.Mixed_5d(x)
        x = self.inception.Mixed_6a(x)
        x = self.inception.Mixed_6b(x)
        x = self.inception.Mixed_6c(x)
        x = self.inception.Mixed_6d(x)
        x = self.inception.Mixed_6e(x)
        x = self.inception.Mixed_7a(x)
        x = self.inception.Mixed_7b(x)
        x = self.inception.Mixed_7c(x)

        # Global average pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))
        features = torch.flatten(x, 1)

        return features

    @torch.no_grad()
    def get_logits(self, x):
        """
        Extract 1000-dimensional logits (for IS).

        Args:
            x: Input images [B, C, H, W] in range [0, 1]
        Returns:
            logits: [B, 1000] class logits
        """
        features = self.get_features(x)
        logits = self.inception.fc(features)
        return logits

    @torch.no_grad()
    def forward(self, x):
        """Default forward returns features."""
        return self.get_features(x)


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Calculate Frechet Distance between two Gaussians."""
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2

    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)

    if not np.isfinite(covmean).all():
        print(f"⚠️  FID calculation: adding {eps} to diagonal of cov estimates")
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = cast(np.ndarray[Any, Any], linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset)))

    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3): # type: ignore
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f"Imaginary component {m}")
        covmean = cast(np.ndarray[Any, Any], covmean.real)

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def calculate_activation_statistics(features):
    """Calculate mean and covariance of features."""
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma


@torch.no_grad()
def compute_fid(real_features, fake_features):
    """Compute Fréchet Inception Distance (FID)."""
    if torch.is_tensor(real_features):
        real_features = real_features.cpu().numpy()
    if torch.is_tensor(fake_features):
        fake_features = fake_features.cpu().numpy()

    mu_real, sigma_real = calculate_activation_statistics(real_features)
    mu_fake, sigma_fake = calculate_activation_statistics(fake_features)

    fid = calculate_frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake)

    return float(fid)


@torch.no_grad()
def compute_inception_score(logits, splits=10, eps=1e-16):
    """
    Compute Inception Score (IS) - FIXED VERSION.

    CRITICAL FIX:
    - Now accepts logits [N, 1000] instead of features [N, 2048]
    - Properly computes class probabilities

    Args:
        logits: [N, 1000] class logits from Inception classifier
        splits: Number of splits for computing std
        eps: Small constant for numerical stability

    Returns:
        mean: Mean IS score
        std: Standard deviation of IS score
    """
    # Convert to tensor if needed
    if not torch.is_tensor(logits):
        logits = torch.from_numpy(logits)

    # Get class probabilities using softmax
    preds = F.softmax(logits, dim=1).cpu().numpy()

    # Verify shape
    if preds.shape[1] != 1000:
        raise ValueError(
            f"Expected 1000 classes, got {preds.shape[1]}. "
            "Make sure you're passing logits, not features!"
        )

    # Split predictions into groups
    split_scores = []
    N = preds.shape[0]

    for i in range(splits):
        part = preds[i * N // splits : (i + 1) * N // splits]

        # Marginal distribution p(y)
        py = np.mean(part, axis=0)

        # KL divergence for each sample: KL(p(y|x) || p(y))
        scores = []
        for j in range(part.shape[0]):
            pyx = part[j]
            kl_div = np.sum(pyx * (np.log(pyx + eps) - np.log(py + eps)))
            scores.append(kl_div)

        split_scores.append(np.exp(np.mean(scores)))

    return float(np.mean(split_scores)), float(np.std(split_scores))


@torch.no_grad()
def compute_kid(real_features, fake_features, num_subsets=100, max_subset_size=1000):
    """Compute Kernel Inception Distance (KID)."""
    if torch.is_tensor(real_features):
        real_features = real_features.cpu().numpy()
    if torch.is_tensor(fake_features):
        fake_features = fake_features.cpu().numpy()

    n = min(real_features.shape[0], fake_features.shape[0], max_subset_size)

    def polynomial_kernel(X, Y):
        """Polynomial kernel k(x,y) = (x^T y / d + 1)^3"""
        d = X.shape[1]
        K = (X @ Y.T / d + 1) ** 3
        return K

    scores = []
    for _ in tqdm(range(num_subsets), desc="Subset"):
        # Random subsets
        idx_real = np.random.choice(real_features.shape[0], n, replace=False)
        idx_fake = np.random.choice(fake_features.shape[0], n, replace=False)

        real_subset = real_features[idx_real]
        fake_subset = fake_features[idx_fake]

        # Compute MMD^2
        K_XX = polynomial_kernel(real_subset, real_subset)
        K_YY = polynomial_kernel(fake_subset, fake_subset)
        K_XY = polynomial_kernel(real_subset, fake_subset)

        mmd2 = (K_XX.sum() - np.diag(K_XX).sum()) / (n * (n - 1))
        mmd2 += (K_YY.sum() - np.diag(K_YY).sum()) / (n * (n - 1))
        mmd2 -= 2 * K_XY.mean()

        scores.append(mmd2)

    return float(np.mean(scores) * 1000), float(np.std(scores) * 1000)



@torch.no_grad()
def extract_features_from_dataloader(
    dataloader,
    feature_extractor,
    max_samples=10000,
    desc="Extracting",
    extract_logits=False,
):
    """
    Extract Inception features (and optionally logits) from a dataloader.

    Args:
        extract_logits: If True, also extract logits for IS computation
    """
    features = []
    logits = [] if extract_logits else None
    total_samples = 0

    feature_extractor.eval()

    pbar = tqdm(
        dataloader,
        desc=desc,
        total=min(len(dataloader), max_samples // dataloader.batch_size),
    )

    for images, _ in pbar:
        if total_samples >= max_samples:
            break

        images = images.to(feature_extractor.device)

        # Normalize to [0, 1]
        images = (images + 1) / 2

        # Extract features
        feats = feature_extractor.get_features(images)
        features.append(feats.cpu())

        # Extract logits if needed
        if extract_logits and logits:
            logs = feature_extractor.get_logits(images)
            logits.append(logs.cpu())

        total_samples += images.shape[0]
        pbar.set_postfix({"samples": total_samples})

    pbar.close()

    features = torch.cat(features, dim=0)[:max_samples]

    if extract_logits:
        logits = torch.cat(logits, dim=0)[:max_samples]
        return features, logits

    return features


@torch.no_grad()
def extract_features_from_generator(
    model,
    feature_extractor,
    num_samples=10000,
    batch_size=64,
    guidance_scale=7.5,
    device="cuda",
    extract_logits=False,
):
    """
    Generate samples and extract features (and optionally logits).
    """
    features = []
    logits = [] if extract_logits else None
    num_batches = (num_samples + batch_size - 1) // batch_size

    model.eval()

    pbar = tqdm(range(num_batches), desc="Generating samples")

    for i in pbar:
        current_batch_size = min(batch_size, num_samples - i * batch_size)

        # Random class labels
        labels = torch.randint(0, 10, (current_batch_size,), device=device)

        # Generate samples
        with torch.autocast(device_type="cuda"):
            samples = model.sample(
                labels=labels,
                shape=(current_batch_size, 1, 32, 32),
                device=device,
                guidance_scale=guidance_scale,
            )

        # Normalize to [0, 1]
        samples = (samples.clamp(-1, 1) + 1) / 2

        # Extract features
        feats = feature_extractor.get_features(samples)
        features.append(feats.cpu())

        # Extract logits if needed
        if extract_logits and logits:
            logs = feature_extractor.get_logits(samples)
            logits.append(logs.cpu())

        # Free memory
        del samples, feats
        if device == "cuda":
            torch.cuda.empty_cache()

    pbar.close()

    features = torch.cat(features, dim=0)

    if extract_logits:
        logits = torch.cat(logits, dim=0)
        return features, logits

    return features


def compute_pairwise_distances(X, Y):
    """
    Compute pairwise Euclidean distances between two feature sets.
    """
    X_norm = (X**2).sum(1).reshape(-1, 1)
    Y_norm = (Y**2).sum(1).reshape(1, -1)
    distances = X_norm + Y_norm - 2.0 * X @ Y.T

    # Numerical stability: ensure non-negative
    distances = torch.clamp(distances, min=0.0)
    distances = torch.sqrt(distances)

    return distances


def compute_nearest_neighbor_distances(X, Y, k=3):
    """
    Compute k-th nearest neighbor distances from X to Y.
    """
    # Compute all pairwise distances
    dist_matrix = compute_pairwise_distances(X, Y)
    kth_distances, _ = torch.kthvalue(dist_matrix, k, dim=1)

    return kth_distances


@torch.no_grad()
def compute_precision_recall(real_features, fake_features, nearest_k=3):
    """
    Compute Precision and Recall metrics for generative models.
    """
    print(f"\n🔍 Computing Precision and Recall (k={nearest_k})...")

    if not torch.is_tensor(real_features):
        real_features = torch.from_numpy(real_features).float()
    if not torch.is_tensor(fake_features):
        fake_features = torch.from_numpy(fake_features).float()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    real_features = real_features.to(device)
    fake_features = fake_features.to(device)

    print(f"Real features: {real_features.shape}")
    print(f"Fake features: {fake_features.shape}")

    print("Computing real manifold radii (k-NN within real data)...")
    real_nn_distances = compute_nearest_neighbor_distances(
        real_features, real_features, k=nearest_k + 1
    )  # k+1 because each point is its own nearest neighbor
    print("Computing fake→real distances...")
    fake_to_real_distances = compute_nearest_neighbor_distances(
        fake_features, real_features, k=1
    )  # 1-NN: closest real sample to each fake sample

    print("Computing real→fake distances...")
    real_to_fake_distances = compute_nearest_neighbor_distances(
        real_features, fake_features, k=1
    )  # 1-NN: closest fake sample to each real sample

    print("Computing precision...")
    fake_nn_real_idx = torch.argmin(
        compute_pairwise_distances(fake_features, real_features), dim=1
    )

    manifold_radii_at_fake = real_nn_distances[fake_nn_real_idx]

    precision = (fake_to_real_distances <= manifold_radii_at_fake).float().mean()

    print("Computing recall...")
    recall = (real_to_fake_distances <= real_nn_distances).float().mean()

    precision = precision.cpu().item()
    recall = recall.cpu().item()

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

    return precision, recall


def compute_density_coverage(real_features, fake_features, nearest_k=3):
    """
    Alternative formulation: Density and Coverage metrics.
    """
    if not torch.is_tensor(real_features):
        real_features = torch.from_numpy(real_features).float()
    if not torch.is_tensor(fake_features):
        fake_features = torch.from_numpy(fake_features).float()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    real_features = real_features.to(device)
    fake_features = fake_features.to(device)

    # Compute k-NN hypersphere radii for real distribution
    real_nn_distances = compute_nearest_neighbor_distances(
        real_features, real_features, k=nearest_k + 1
    )

    # Density: proportion of fake samples within real hyperspheres
    fake_to_real_nn = compute_nearest_neighbor_distances(
        fake_features, real_features, k=1
    )

    fake_nn_real_idx = torch.argmin(
        compute_pairwise_distances(fake_features, real_features), dim=1
    )
    manifold_radii = real_nn_distances[fake_nn_real_idx]

    density = (fake_to_real_nn <= manifold_radii).float().mean().cpu().item()

    # Coverage: proportion of real hyperspheres that contain at least one fake sample
    dist_matrix = compute_pairwise_distances(real_features, fake_features)
    min_distances = torch.min(dist_matrix, dim=1)[0]

    coverage = (min_distances <= real_nn_distances).float().mean().cpu().item()

    return density, coverage

def evaluate_metrics_with_precision_recall(
    model,
    real_dataloader,
    device="cuda",
    guidance_scale=7.5,
    num_samples=10000,
    save_dir="./evaluation",
    nearest_k=3,
):
    """
    Comprehensive evaluation: FID, IS, KID, Precision, Recall + sample generation.
    """
    os.makedirs(save_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print("📊 EVALUATING GENERATIVE MODEL METRICS (WITH PRECISION/RECALL)")
    print("=" * 70)

    # Initialize feature extractor
    print("\n🔧 Loading Inception V3 feature extractor...")
    feature_extractor = InceptionV3FeatureExtractor(device=device, use_fp16=False)

    # Extract real features
    print(f"\n📸 Extracting features from {num_samples} real images...")
    real_features = extract_features_from_dataloader(
        real_dataloader,
        feature_extractor,
        max_samples=num_samples,
        desc="Real images",
        extract_logits=False,
    )

    # Extract fake features AND logits
    print(
        f"\n🎨 Generating {num_samples} fake images and extracting features + logits..."
    )
    fake_features, fake_logits = extract_features_from_generator(
        model,
        feature_extractor,
        num_samples=num_samples,
        batch_size=128,
        guidance_scale=guidance_scale,
        device=device,
        extract_logits=True,
    )

    # Compute standard metrics
    print("\n🧮 Computing standard metrics...")

    print("Computing FID...")
    fid_score = compute_fid(real_features, fake_features)
    print(f"FID score: {fid_score:.4f}")

    print("Computing Inception Score...")
    is_mean, is_std = compute_inception_score(fake_logits)
    print(f"IS score: {is_mean:.4f} ± {is_std:.4f}")

    print("Computing KID...")
    kid_mean, kid_std = compute_kid(real_features, fake_features)
    print(f"KID score: {kid_mean:.4f} ± {kid_std:.4f}")

    # Compute Precision and Recall
    print("\n🎯 Computing Precision and Recall...")
    precision, recall = compute_precision_recall(
        real_features, fake_features, nearest_k=nearest_k
    )

    # Optional: Compute Density and Coverage (alternative formulation)
    print("\n📈 Computing Density and Coverage (alternative metrics)...")
    density, coverage = compute_density_coverage(
        real_features, fake_features, nearest_k=nearest_k
    )

    # Print results
    print("\n" + "=" * 70)
    print("📈 RESULTS")
    print("=" * 70)
    print(f"  FID Score:         {fid_score:.4f}  (lower is better, good < 50)")
    print(f"  Inception Score:   {is_mean:.4f} ± {is_std:.4f}  (higher is better)")
    print(f"  KID Score:         {kid_mean:.4f} ± {kid_std:.4f}  (lower is better)")
    print(
        f"\n  Precision:         {precision:.4f}  (quality/fidelity, higher is better)"
    )
    print(f"  Recall:            {recall:.4f}  (diversity/coverage, higher is better)")
    print(f"\n  Density:           {density:.4f}  (alternative to precision)")
    print(f"  Coverage:          {coverage:.4f}  (alternative to recall)")
    print("=" * 70 + "\n")

    # Interpretation guide
    print("💡 INTERPRETATION GUIDE:")
    print("  • High Precision + High Recall = Excellent model (realistic & diverse)")
    print(
        "  • High Precision + Low Recall = Mode collapse (realistic but limited variety)"
    )
    print("  • Low Precision + High Recall = Poor quality (diverse but unrealistic)")
    print("  • Low Precision + Low Recall = Poor model overall")
    print("=" * 70 + "\n")

    # Save results to JSON
    import json

    results = {
        "fid": float(fid_score),
        "inception_score_mean": float(is_mean),
        "inception_score_std": float(is_std),
        "kid_mean": float(kid_mean),
        "kid_std": float(kid_std),
        "precision": float(precision),
        "recall": float(recall),
        "density": float(density),
        "coverage": float(coverage),
        "guidance_scale": guidance_scale,
        "num_samples": num_samples,
        "nearest_k": nearest_k,
    }

    results_path = os.path.join(save_dir, "metrics_with_precision_recall.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"💾 Results saved to: {results_path}")

    return results
