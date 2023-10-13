import torch

def check_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device

if __name__ == "__main__":
    device = check_device()
    print(f"YOLO is using: {device}")
