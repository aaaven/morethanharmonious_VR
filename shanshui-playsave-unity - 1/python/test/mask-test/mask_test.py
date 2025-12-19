import numpy as np
import cv2

def draw_bezier_curve(mask, p0, p1, p2, steps=100):
    """在mask上绘制贝塞尔曲线"""
    for t in np.linspace(0, 1, steps):
        x = int((1-t)**2 * p0[0] + 2*(1-t)*t * p1[0] + t**2 * p2[0])
        y = int((1-t)**2 * p0[1] + 2*(1-t)*t * p1[1] + t**2 * p2[1])
        cv2.circle(mask, (x, y), 1, 255, -1)

def draw_tangent_and_fill_with_bezier(mask, x1, y1, r1, x2, y2, r2):
    """用贝塞尔曲线连接两个圆"""
    # 计算两个圆之间的向量
    vec_x = x2 - x1
    vec_y = y2 - y1
    dist = np.sqrt(vec_x**2 + vec_y**2)
    
    # 确保两个圆之间有一定的距离
    if dist < r1 + r2:
        dist = r1 + r2
    
    # 计算单位向量
    unit_vec_x = vec_x / dist
    unit_vec_y = vec_y / dist
    
    # 计算切点
    p0 = (int(x1 + r1 * unit_vec_x), int(y1 + r1 * unit_vec_y))
    p2 = (int(x2 - r2 * unit_vec_x), int(y2 - r2 * unit_vec_y))
    
    # 控制点（贝塞尔曲线的顶点），根据圆的大小调整控制点位置
    mid_x = (p0[0] + p2[0]) // 2
    mid_y = (p0[1] + p2[1]) // 2
    control_x = mid_x - int(unit_vec_y * min(r1, r2) * 0.8)  # 调整控制点位置
    control_y = mid_y + int(unit_vec_x * min(r1, r2) * 0.8)  # 调整控制点位置
    
    # 绘制贝塞尔曲线
    draw_bezier_curve(mask, p0, (control_x, control_y), p2)

    # 绘制两个圆
    cv2.circle(mask, (x1, y1), r1, 255, -1)
    cv2.circle(mask, (x2, y2), r2, 255, -1)

# 创建一个空的遮罩
mask = np.zeros((512, 512), dtype=np.uint8)

# 示例圆心坐标和半径
x1, y1, r1 = 150, 250, 50
x2, y2, r2 = 350, 250, 75

# 调用函数绘制贝塞尔曲线和圆
draw_tangent_and_fill_with_bezier(mask, x1, y1, r1, x2, y2, r2)

# 显示结果
cv2.imshow("Mask with Bezier Curve", mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
