import os
import cv2
import torch
import numpy as np
from diffusers import StableDiffusionXLInpaintPipeline
from PIL import Image
import random
from torch.nn import functional as F
import warnings
import time
from train_log.RIFE_HDv3 import Model
from concurrent.futures import ThreadPoolExecutor, as_completed
import socket
import struct
from PIL import ImageFilter, ImageOps

warnings.filterwarnings("ignore")

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

print(f"Using device: {device}")

# 加载模型
pipe = StableDiffusionXLInpaintPipeline.from_pretrained(r"D:\linz\local_run\sdxl-turbo", safety_checker=None)
pipe = pipe.to(device)

# 加载 RIFE 模型
model = Model()
model.load_model('train_log', -1)  # 请确保提供正确的模型路径
model.eval()
model.device()

# 定义图片大小调整函数
def resize(image, target_width, target_height):
    return image.resize((target_width, target_height))

def crop_to_square(image, mask, size=512):
    mask = mask.convert('L')
    mask_array = np.array(mask)

    # 对mask进行卷积操作，快速找到512x512区域的白色像素总和
    conv_mask = cv2.filter2D(mask_array, -1, np.ones((size, size), np.float32))

    # 找到卷积结果中最大值的索引
    y, x = np.unravel_index(np.argmax(conv_mask), conv_mask.shape)

    # 确保裁剪区域不会超出图像边界
    x = min(x, mask_array.shape[1] - size)
    y = min(y, mask_array.shape[0] - size)

    # 裁剪图像和mask
    cropped_image = image.crop((x, y, x + size, y + size))
    cropped_mask = mask.crop((x, y, x + size, y + size))

    return cropped_image, cropped_mask, x, y



# 定义图像到图像的转换函数
def img2img(src, mask, prompt, strength, guidance_scale):
    print(f"Starting img2img with prompt: {prompt}")
    result = pipe(prompt=prompt, image=src, mask_image=mask, num_inference_steps=4, strength=strength, guidance_scale=guidance_scale).images[0]
    print("Completed img2img")
    return result

# 合并图片，确保精确控制回贴位置
def merge_images(ground, img2img_result, mask, x, y):
    mask = mask.convert('1').resize(img2img_result.size)
    canvas = Image.new("RGB", ground.size)
    canvas.paste(ground, (0, 0))
    canvas.paste(img2img_result, (x, y), mask)
    return canvas

# 定义一个线程池
executor = ThreadPoolExecutor(max_workers=1)

# 生成插帧图像并通过Socket传输
def generate_interpolated_frames(img0, img1, exp=5):
    # 转换图片为张量
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
    print("Interpolated frames generated")

    # 将张量转换回图像列表
    images = [(img[0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w] for img in img_list]
    images = [Image.fromarray(img.astype(np.uint8)) for img in images]
    return images

def send_images(images, conn):
    for index, image in enumerate(images):
        data = image.tobytes()
        conn.sendall(data)
        print(f"Frame {index + 1}/{len(images)} sent.")
        
        # 等待客户端的确认
        response = conn.recv(2)
        if response.decode('utf-8') != "OK":
            print("Error: did not receive OK from client.")
            break

def main(host='127.0.0.1', port=12345):
    mask_dir = "D:\\linz\\test\\mask"
    processed_masks = set()
    mask_index = 0

    scale = 0.5

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, port))
        s.listen()
        print(f"Server listening on {host}:{port}")

        conn, addr = s.accept()

        
        with conn:
            print(f"Connected by {addr}")
            # 打开地面图像并调整大小
            ground_img = Image.open("ground.jpg")
            ground_img = ground_img.resize((int(ground_img.width * scale), int(ground_img.height* scale)))
            print(ground_img.width, ground_img.height)
            # 初始输入图像
            input_image = ground_img
            merged_images = []
            merged_images.append(input_image)
            send_images(merged_images, conn)

            prompts = [
            "Landfill. Pollution is the introduction of harmful materials into the environment. Landfills collect garbage and other land pollution in a central location. Many places are running out of space for landfills",
            "City Light Pollution. Boats, buildings, street lights, and even fireworks contribute to the light pollution in Victoria Harbor, Hong Kong. Light pollution can be detrimental to the health of people and animals in the area",
            "Human induced oil spills devastating wildlife and resulting in animals becoming sick or dying ",
            "Human induced disasters, such as oil spills, can be devastating for all forms of wildlife. Often times resulting in animals becoming sick or dying.",
            "Wildfires scorch the land in Malibu Creek State Park. As the wind picks up, the fire begins to spread faster",
            "The tallest towers of Shanghai, China, rise above the haze. Shanghai's smog is a mixture of pollution from coal, the primary source of energy for most homes and businesses in the region, as well as emissions from vehicles"
            ]   

            while True:
                mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.png')])

                if mask_index < len(mask_files):
                    mask_path = os.path.join(mask_dir, mask_files[mask_index])
                    print(f"Processing {mask_path}")

                    # 读取mask图像
                    mask_img = Image.open(mask_path).convert('L')
                    mask_img = mask_img.resize((input_image.width, input_image.height))

                    # 生成图像到图像的结果
                    prompt = random.choice(prompts)
                    strength = 0.75
                    guidance_scale = 1.5

                    # 裁剪图像到正方形
                    square_ground, square_mask, x, y = crop_to_square(input_image, mask_img, 512)

                    # 图像到图像的转换
                    img2img_result = img2img(square_ground, square_mask, prompt, strength, guidance_scale)

                    images = generate_interpolated_frames(square_ground, img2img_result, exp=4)

                    # 将插帧图像逐一合并回原始图像并生成完整图像序列
                    merged_images = []
                    for interpolated_image in images:
                        final_image = merge_images(input_image, interpolated_image, square_mask, x, y)
                        merged_images.append(final_image)

                    send_images(merged_images, conn)

                    # 更新输入图像为最新生成的图像
                    input_image = final_image

                    mask_index += 1

                time.sleep(0.1)
                    
                
        print("Server will close now.")

    


if __name__ == "__main__":
    main()
