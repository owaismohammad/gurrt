import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))

    # Simple GPU computation
    x = torch.rand(1000, 1000).cuda()
    y = torch.mm(x, x)
    print("GPU computation successful ✅")
else:
    print("GPU not available ❌")
