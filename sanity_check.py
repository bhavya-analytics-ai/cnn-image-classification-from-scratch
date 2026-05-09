import torch

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print("Device:", DEVICE)

x = torch.randn(1, 3, 224, 224).to(DEVICE)
print("Tensor shape:", x.shape)
print("Tensor device:", x.device)
