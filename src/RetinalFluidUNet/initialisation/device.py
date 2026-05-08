import torch

print("Device initialisation started.")

print(f"Torch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Torch CUDA version: {torch.version.cuda}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device == torch.device("cpu"):
    print("No GPU available, using CPU.")
    raise NotImplementedError("No GPU available, using CPU.")
