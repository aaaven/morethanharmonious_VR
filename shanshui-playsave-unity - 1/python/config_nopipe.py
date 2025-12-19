import torch
from diffusers import StableDiffusionXLInpaintPipeline
from train_log.RIFE_HDv3 import Model

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

print(f"Using device: {device}")


# 加载 RIFE 模型
model = Model()
model.load_model('train_log', -1)  # 请确保提供正确的模型路径
model.eval()
model.device()
