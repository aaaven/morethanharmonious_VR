import cv2
import numpy as np
import time
# 加载背景图像和掩码图像
background = cv2.imread('/image/background.png')
mask = cv2.imread('/image/mask.jpg', cv2.IMREAD_GRAYSCALE)

# 确保背景图像和掩码图像大小一致
height, width = background.shape[:2]
mask = cv2.resize(mask, (width, height))

# 为图像创建网格
x, y = np.meshgrid(np.arange(width), np.arange(height))

# 定义变形参数
amplitude = 20
frequency = 2
spatial_scale = 150

time0 = time.time()
# 动态生成扭曲效果
for t in range(60):  # 生成60帧动画
    phase_shift = t / 10.0  # 动态调整相位以产生动画效果
    
    distortion_x = amplitude * np.sin(frequency * np.pi * y / spatial_scale + phase_shift)
    distortion_y = amplitude * np.cos(frequency * np.pi * x / spatial_scale + phase_shift)

    x_new = np.clip(x + distortion_x.astype(np.float32), 0, width - 1)
    y_new = np.clip(y + distortion_y.astype(np.float32), 0, height - 1)

    map_xy = np.dstack((x_new, y_new)).astype(np.float32)
    distorted_image = cv2.remap(background, map_xy, None, interpolation=cv2.INTER_LINEAR)

    # 创建一个三通道掩码（因为distorted_image是彩色图像）
    mask_3d = np.stack([mask]*3, axis=-1)

    # 将原背景图像和变形后的图像结合
    result_image = background.copy()
    result_image[mask_3d > 0] = distorted_image[mask_3d > 0]

    print(time.time() - time0)
    time0 = time.time()
    # 显示结果图像
    cv2.imshow('Result Image', result_image)
    if cv2.waitKey(100) & 0xFF == ord('q'):  # 每帧停留100ms
        break

cv2.destroyAllWindows()
