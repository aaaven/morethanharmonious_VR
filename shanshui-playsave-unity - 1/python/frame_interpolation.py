import torch
import numpy as np
from PIL import Image
from torch.nn import functional as F
from config_nopipe import model, device

def generate_interpolated_frames(img0, img1, exp=4):
    img0_tensor = (torch.tensor(np.array(img0).transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)
    img1_tensor = (torch.tensor(np.array(img1).transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)

    n, c, h, w = img0_tensor.shape
    ph = ((h - 1) // 32 + 1) * 32
    pw = ((w - 1) // 32 + 1) * 32
    padding = (0, pw - w, 0, ph - h)
    img0_tensor = F.pad(img0_tensor, padding)
    img1_tensor = F.pad(img1_tensor, padding)

    img_list = [img0_tensor, img1_tensor]
    for _ in range(exp):
        tmp = []
        for j in range(len(img_list) - 1):
            mid = model.inference(img_list[j], img_list[j + 1])
            tmp.append(img_list[j])
            tmp.append(mid)
        tmp.append(img1_tensor)
        img_list = tmp

    images = [(img[0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w] for img in img_list]
    images = [Image.fromarray(img.astype(np.uint8)) for img in images]
    return images
