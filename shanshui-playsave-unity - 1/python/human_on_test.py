import socket

def start_server():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(('127.0.0.1', 8080))  # 绑定IP地址和端口号
    server_socket.listen(1)
    
    print("等待连接...")
    client_socket, addr = server_socket.accept()
    print(f"连接来自: {addr}")
    
    while True:
        data = client_socket.recv(1024).decode('utf-8')
        if not data:
            break
        print(f"接收到的信息: {data}")

    client_socket.close()
    server_socket.close()

if __name__ == "__main__":
    start_server()
