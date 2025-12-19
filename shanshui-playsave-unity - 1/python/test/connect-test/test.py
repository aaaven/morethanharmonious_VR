import socket
import numpy as np
import threading
import time
import cv2

# 配置
HOST = '0.0.0.0'  # 监听所有可用的IP地址
PORT = 12346  # 与Unity端口匹配
WIDTH = 1047
HEIGHT = 1544
CIRCLE_RADIUS = 10  # 圆的半径
CROP_SIZE = 512  # 截取区域的大小

# 队列用于存储接收到的坐标点
xy_queue = []

# 锁定队列访问
lock = threading.Lock()

def handle_client_connection(client_socket):
    with client_socket:
        while True:
            data = client_socket.recv(1024)
            if not data:
                break
            data_str = data.decode('utf-8').strip()
            for point in data_str.splitlines():
                x, y = map(int, point.split(','))
                # 将y轴颠倒
                y = HEIGHT - y
                with lock:
                    xy_queue.append((x, y))

def start_server():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((HOST, PORT))
    server.listen(5)
    print(f"Listening on {HOST}:{PORT}")

    while True:
        client_sock, addr = server.accept()
        print(f"Accepted connection from {addr}")
        client_handler = threading.Thread(
            target=handle_client_connection,
            args=(client_sock,)
        )
        client_handler.start()

def generate_mask_image():
    while True:
        time.sleep(2)
        with lock:
            if xy_queue:
                # 创建黑色背景图像
                mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
                
                # 找到左上角点
                min_x = min(x for x, y in xy_queue)
                min_y = min(y for x, y in xy_queue)
                
                # 确保截取区域在图像范围内
                start_x = max(0, min(min_x, WIDTH - CROP_SIZE))
                start_y = max(0, min(min_y, HEIGHT - CROP_SIZE))
                
                # 截取512x512区域内的点并绘制圆
                cropped_points = [(x, y) for x, y in xy_queue 
                                  if start_x + CIRCLE_RADIUS <= x < start_x + CROP_SIZE - CIRCLE_RADIUS and 
                                     start_y + CIRCLE_RADIUS <= y < start_y + CROP_SIZE - CIRCLE_RADIUS]

                for x, y in cropped_points:
                    cv2.circle(mask, (x, y), CIRCLE_RADIUS, 200, -1)  # -1表示填充整个圆圈

                # 保存图像
                filename = f"mask_{int(time.time())}.png"
                cv2.imwrite(filename, mask)
                print(f"Saved mask image as {filename}")
                xy_queue.clear()

if __name__ == "__main__":
    # 启动服务器线程
    server_thread = threading.Thread(target=start_server)
    server_thread.start()

    # 启动生成mask的线程
    generate_mask_thread = threading.Thread(target=generate_mask_image)
    generate_mask_thread.start()
