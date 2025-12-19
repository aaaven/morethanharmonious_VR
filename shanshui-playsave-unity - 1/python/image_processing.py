import numpy as np
from PIL import Image
import cv2

def resize(image, target_width, target_height):
    return image.resize((target_width, target_height))

def merge_images(ground, img2img_result, mask, x, y):
    mask = mask.convert('L').resize(img2img_result.size)  # 转换为灰度图像（L模式）
    
    # 将ground中对应区域剪裁出来，用于与img2img_result进行混合
    ground_crop = ground.crop((x, y, x + img2img_result.width, y + img2img_result.height))
    
    # 根据mask的灰度值进行混合
    blended = Image.composite(img2img_result, ground_crop, mask)
    
    # 在canvas上创建并粘贴最终的图像
    canvas = Image.new("RGB", ground.size)
    canvas.paste(ground, (0, 0))
    canvas.paste(blended, (x, y))
    
    return canvas

def draw_tangent_and_fill(mask, x1, y1, r1, x2, y2, r2):
    # 计算圆心之间的距离
    dist = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    # 如果距离为0，则直接返回，避免NaN计算
    if dist == 0:
        return

    # 向量方向
    vec_x = x2 - x1
    vec_y = y2 - y1

    norm_vec = np.sqrt(vec_x ** 2 + vec_y ** 2)
    if norm_vec == 0:
        return  # 避免除以0

    # 单位向量
    unit_vec_x = vec_x / norm_vec
    unit_vec_y = vec_y / norm_vec

    # 第一组切点
    tx1_1 = x1 + r1 * unit_vec_x
    ty1_1 = y1 + r1 * unit_vec_y
    tx2_1 = x2 - r2 * unit_vec_x
    ty2_1 = y2 - r2 * unit_vec_y

    # 第二组切点（相反方向）
    tx1_2 = x1 - r1 * unit_vec_x
    ty1_2 = y1 - r1 * unit_vec_y
    tx2_2 = x2 + r2 * unit_vec_x
    ty2_2 = y2 + r2 * unit_vec_y

    # 确保所有点都是有效的
    if not (np.isnan(tx1_1) or np.isnan(ty1_1) or np.isnan(tx2_1) or np.isnan(ty2_1) or
            np.isnan(tx1_2) or np.isnan(ty1_2) or np.isnan(tx2_2) or np.isnan(ty2_2)):

        # 填充两个圆之间的区域（两边）
        pts1 = np.array([[int(tx1_1), int(ty1_1)], [int(x1 + r1 * unit_vec_y), int(y1 - r1 * unit_vec_x)],
                         [int(x2 + r2 * unit_vec_y), int(y2 - r2 * unit_vec_x)], [int(tx2_1), int(ty2_1)]], np.int32)
        cv2.fillConvexPoly(mask, pts1, 255)

        pts2 = np.array([[int(tx1_2), int(ty1_2)], [int(x1 - r1 * unit_vec_y), int(y1 + r1 * unit_vec_x)],
                         [int(x2 - r2 * unit_vec_y), int(y2 + r2 * unit_vec_x)], [int(tx2_2), int(ty2_2)]], np.int32)
        cv2.fillConvexPoly(mask, pts2, 255)

        # 绘制两个圆
        cv2.circle(mask, (x1, y1), r1, 255, -1)
        cv2.circle(mask, (x2, y2), r2, 255, -1)
        mask_blurred = cv2.GaussianBlur(mask, (11, 11), 0)
        np.copyto(mask, mask_blurred)