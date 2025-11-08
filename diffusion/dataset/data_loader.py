from torchvision import datasets, transforms
from torch.utils.data import DataLoader


FASHION_CLASSES = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot"
}

def get_fashion_mnist_dataloader(batch_size=32, root='./data', download=True):
    """
    Load Fashion-MNIST dataset.
    """
    transform = transforms.Compose([
        transforms.Resize(32),  # Resize from 28x28 to 32x32
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # Scale to [-1, 1] for diffusion
    ])

    # Load Fashion-MNIST
    train_dataset = datasets.FashionMNIST(
        root=root,
        train=True,
        download=download,
        transform=transform
    )

    test_dataset = datasets.FashionMNIST(
        root=root,
        train=False,
        download=download,
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    print(f"✅ Loaded Fashion-MNIST: {len(train_dataset)} training, {len(test_dataset)} test samples")
    return train_dataset, test_dataset, train_loader, test_loader
