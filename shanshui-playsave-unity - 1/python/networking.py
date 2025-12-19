import socket
from PIL import Image

def send_image(image, conn):
    """发送单张图像并等待客户端确认"""
    data = image.tobytes()
    conn.sendall(data)

    response = conn.recv(2)
    if response.decode('utf-8') != "OK":
        print("Error: did not receive OK from client.")
