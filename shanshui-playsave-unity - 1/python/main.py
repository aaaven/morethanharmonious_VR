import socket
import threading
import os
import time
import keyboard
import cv2
import numpy as np
from PIL import Image
from moviepy.editor import ImageSequenceClip
import torch
import random
from image_processing import resize, merge_images, draw_tangent_and_fill
from frame_interpolation import generate_interpolated_frames
from networking import send_image
from pipeline import prompt_embeds, pooled_prompt_embeds, pipe_reduced, aug_embs, dtype
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# pipe_reduced = torch.compile(pipe_reduced)
# pipe_reduced = torch.compile(pipe_reduced, mode="reduce-overhead", fullgraph=True, dynamic=False)
# print("all reduce-overhead fullgraph static")
def img2img(img, mask, i):
    image = torch.tensor(np.array(img)[None, ...].transpose((0, 3, 1, 2)), device=device, dtype=torch.bfloat16) / 255.0
    mask = torch.tensor(np.array(mask.resize((512, 512)), dtype=np.bool_)[None, None], device=device, dtype=dtype)
    with torch.inference_mode():
        img2img_result = pipe_reduced(
            prompt_embeds=prompt_embeds[i],
            aug_emb=aug_embs[i],
            image=image,
            mask=mask,
        )
    return Image.fromarray((img2img_result[0] * 255).to('cpu', dtype=torch.float32).numpy().transpose(1, 2, 0).astype(np.uint8))

# Setup
HOST = '0.0.0.0'
PORT = 12346
WIDTH = 1047
HEIGHT = 1544
CIRCLE_RADIUS = 65
SPEED_REDUCE = 0.2
CROP_SIZE = 512
DILATE_KERNEL_SIZE = 20
CIRCLE_NUM = 20
GAUSSIAN_KERNEL_SIZE = 41  # 高斯模糊核的大小
BLUR_RADIUS = (GAUSSIAN_KERNEL_SIZE + DILATE_KERNEL_SIZE) // 2  # 核大小的一半，作为模糊半径

prompts_list = [
    "Abandoned Chernobyl amusement park with a rusted Ferris wheel against a radioactive sky.",
"Exxon Valdez oil spill covering the waters of Prince William Sound, with distressed wildlife.",
"Dense haze from Southeast Asian forest fires, with obscured sun and masked city residents.",
"European city under a scorching sun during the 2003 heat wave, streets deserted and hot.",
"Ruins of the Fukushima nuclear plant post-tsunami, under a cloudy sky with radioactive symbols.",
"California forest engulfed in flames at sunset, with firefighters battling the intense wildfire.",
"Stark contrast of lush Amazon rainforest and adjacent deforested barren land with stumps.",
"Polar bear on a melting ice fragment in the Arctic, surrounded by water and distant icebergs.",
"Australian bushfires scene with fleeing kangaroos and a landscape engulfed in red flames.",
"Bleached coral in the Great Barrier Reef, with vibrant living coral and swimming small fish.",
"Sea turtle navigating through ocean cluttered with plastic debris, near a shadowy city skyline.",
"Brazilian Amazon in flames, with rising smoke depicting rainforest destruction.",
"Australian bushfires from above, showing fire consuming forests and causing wildlife distress.",
"California's scorched earth and barren landscapes with wildfires and smoke clouds.",
"East African farmlands overrun by swarms of locusts, devastating crops and causing despair.",
]

class Server:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.xy_queue = []
        self.latest_mask = None
        self.latest_mask_info = None
        self.images_to_send = []
        self.image_count = 0
        self.input_image = None  # 用于存储当前的输入图像
        self.images_to_save = []
        self.images_to_show = []
        self.history_dir = "history"
        self.distortion_active = False
        self.prompt_idx = None
        self.reverse_show_images = False
        self.reset_event = threading.Event()
        self.running = True
        self.hmd_status = "HMD Unmounted"
        self.archiving = False


    def hmd_status_listener(self, host, port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((host, port))
            s.listen(1)
            print(f"HMD status server listening on {host}:{port}")
            conn, addr = s.accept()
            with conn:
                print(f"Connected by {addr}")
                while self.running:
                    data = conn.recv(1024)
                    if not data:
                        break
                    hmd_status = data.decode('utf-8').strip()
                    print(hmd_status)
                    if hmd_status == "HMD Unmounted":
                        self.hmd_status = "HMD Unmounted"
                        self.reset_system()
                    elif hmd_status == "HMD Mounted":
                        self.hmd_status = "HMD Mounted"
                        self.reset_event.set()

    def keyboard_listener(self):
        while self.running:
            if keyboard.is_pressed('r'):  # 检查是否按下 'r' 键
                print("Reset key pressed. Resetting system...")
                self.reset_system()
                
            time.sleep(0.1)  # 避免占用过多CPU，适当设置监听的间隔时间

    # 重置系统的函数
    def reset_system(self):
        ground_img = Image.open("test/image/ground.jpg")
        scale = 0.5
        self.input_image = ground_img.resize((int(ground_img.width * scale), int(ground_img.height * scale)))
        self.latest_mask = None
        self.latest_mask_info = None
        self.image_count = 0
        self.images_to_send.clear()
        self.reverse_show_images = True
        self.reset_event.clear()
        self.show_finish = True
        time.sleep(2)
        self.input_image = ground_img.resize((int(ground_img.width * scale), int(ground_img.height * scale)))
        self.images_to_send.append(self.input_image)
        # # 保证开始时有一帧
        # self.images_to_show.clear()
        # self.images_to_show.append(cv2.cvtColor(np.array(self.input_image), cv2.COLOR_RGB2BGR))  

    
    def save_images_to_new_video(self, fps=10):
        if not self.images_to_show:
            return
        
        self.show_finish = False

        # 确定下一个视频的文件名
        video_files = [f for f in os.listdir(self.history_dir) if f.endswith('.mp4')]
        next_video_idx = len(video_files)
        next_video_path = os.path.join(self.history_dir, f"{next_video_idx}.mp4")

        # 将图像从BGR格式转换回RGB格式（MoviePy需要图像为RGB格式）
        images_rgb = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in self.images_to_show]

        # 使用MoviePy创建视频
        clip = ImageSequenceClip(images_rgb, fps=fps)
        clip.write_videofile(next_video_path, codec='libx264')

    def handle_client_connection(self, client_socket):
        with client_socket:
            while True:
                if self.archiving:
                    # 丢弃输入并可选发送提示
                    # client_socket.sendall(b'ARCHIVING')  # 如果需要通知前端
                    continue
                data = client_socket.recv(1024)
                if not data:
                    break
                data_str = data.decode('utf-8').strip()
                for point in data_str.splitlines():
                    parts = point.split(',')
                    if len(parts) != 3:
                        continue  # 跳过不符合预期的数据
                    
                    try:
                        x, y, speed = map(float, parts)
                        self.xy_queue.append((x, y, speed))
                    except ValueError:
                        continue  # 跳过无法转换为浮点数的数据


    def start_server(self):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind((HOST, PORT))
        server.listen(5)
        print(f"Listening on {HOST}:{PORT}")

        while True:
            client_sock, addr = server.accept()
            print(f"Accepted connection from {addr}")
            client_handler = threading.Thread(
                target=self.handle_client_connection,
                args=(client_sock,)
            )
            client_handler.start()

    def generate_mask_image_periodically(self):
        while True:
            time.sleep(1)
            if self.xy_queue:
                self._generate_and_update_mask()

    def _generate_and_update_mask(self):
        if not self.xy_queue:
            return

        xy_array = np.array(self.xy_queue)

        center_x = int(np.mean([np.min(xy_array[:, 0]), np.max(xy_array[:, 0])]))
        center_y = int(np.mean([np.min(xy_array[:, 1]), np.max(xy_array[:, 1])]))

        start_x = max(0, min(center_x - 256, WIDTH - 512))
        start_y = max(0, min(center_y - 256, HEIGHT - 512))
        end_x = start_x + 512
        end_y = start_y + 512

        cropped_mask = np.zeros((512, 512), dtype=np.uint8)

        valid_points = xy_array[
            (xy_array[:, 0] >= start_x + CIRCLE_RADIUS + BLUR_RADIUS) &
            (xy_array[:, 0] < end_x - CIRCLE_RADIUS - BLUR_RADIUS) &
            (xy_array[:, 1] >= start_y + CIRCLE_RADIUS + BLUR_RADIUS) &
            (xy_array[:, 1] < end_y - CIRCLE_RADIUS - BLUR_RADIUS)
        ]
        relative_points = valid_points[:, :2] - [start_x, start_y]

        i = 0
        while i < len(relative_points) - 1:
            x1, y1 = relative_points[i].astype(int)
            r1 = int(max(CIRCLE_RADIUS - valid_points[i, 2] * 0.02, 15))

            for j in range(i + 1, len(relative_points)):
                x2, y2 = relative_points[j].astype(int)
                r2 = int(max(CIRCLE_RADIUS - valid_points[j, 2] * 0.02, 15))

                dist = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

                if dist + r2 <= r1/2:
                    continue
                else:
                    draw_tangent_and_fill(cropped_mask, x1, y1, r1, x2, y2, r2)
                    i = j - 1 
                    break
            i += 1

        # 形态学处理
        kernel = np.ones((DILATE_KERNEL_SIZE, DILATE_KERNEL_SIZE), np.uint8)
        continuous_mask = cv2.dilate(cropped_mask, kernel, iterations=1)

        # 高斯模糊处理
        blurred_mask = cv2.GaussianBlur(continuous_mask, (GAUSSIAN_KERNEL_SIZE, GAUSSIAN_KERNEL_SIZE), 0)

        cv2.imshow("Heatmap", blurred_mask)
        cv2.waitKey(1)

        self.xy_queue.clear()
        self.latest_mask = cropped_mask
        self.latest_mask_info = (start_x, start_y)

    def send_images_thread(self, conn):
        while True:
            if self.images_to_send:
                image = self.images_to_send.pop(0)

                send_image(image, conn)
            time.sleep(0.008)

    def save_images_thread(self):
        while True:
            if self.images_to_save:
                image = self.images_to_save.pop(0)  # 从 images_to_save 列表中取出图片
                file_path = os.path.join(self.history_dir, f"image_{self.image_count}.png")
                image.save(file_path)
                print(f"Image saved to {file_path}")
                self.image_count += 1  # 每保存一张图片后增加计数
            time.sleep(0.0001)  # 适当延长等待时间以应对较慢的存储速度

    def show_images_thread(self):
        current_img = None
        
        cv2.namedWindow('Image Display', cv2.WINDOW_NORMAL)
        cv2.moveWindow('Image Display', 0, -1080) 
        cv2.setWindowProperty('Image Display', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        video_index = 0
        reverse_index = 1 

        while True:
            # 动态获取当前的所有视频文件
            video_files = sorted([f for f in os.listdir(self.history_dir) if f.endswith('.mp4')])

            if not video_files:
                time.sleep(1)  # 如果没有视频文件，等待1秒后重试
                continue

            # 确保video_index在范围内
            video_index = video_index % len(video_files)

            # 播放当前视频文件
            video_path = os.path.join(self.history_dir, video_files[video_index])
            capture = cv2.VideoCapture(video_path)
            
            frames = []  # 存储所有帧以备反向播放使用

            while capture.isOpened():
                ret, frame = capture.read()
                if not ret:
                    break

                frames.append(frame)
                start_time = time.time()
                # 判断是否需要反向展示
                if self.reverse_show_images and self.images_to_show:
                    if reverse_index == 1:
                        time.sleep(2)
                        # 初始化 VideoWriter
                        video_files = [f for f in os.listdir(self.history_dir) if f.endswith('.mp4')]
                        next_video_idx = len(video_files)
                        next_video_path = os.path.join(self.history_dir, f"{next_video_idx}.mp4")
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        img_for_shape = self.images_to_show[-1]
                        height, width = img_for_shape.shape[:2]
                        print(f"[INFO] Start Saving Video to {next_video_path}")
                        self.video_writer = cv2.VideoWriter(next_video_path, fourcc, 10, (width, height))
                        self._progress_total = len(self.images_to_show)
                        self._progress_count = 0
                        self.archiving = True
                    reverse_index = reverse_index % len(self.images_to_show)
                    current_img = self.images_to_show[-reverse_index - 1]  # 从最后一帧开始反向播放

                    # 写入当前帧到视频
                    if hasattr(self, 'video_writer') and current_img is not None:
                        self.video_writer.write(current_img)
                        self._progress_count += 1
                        # 打印进度条
                        bar_len = 30
                        filled_len = int(round(bar_len * self._progress_count / float(self._progress_total)))
                        percents = round(100.0 * self._progress_count / float(self._progress_total), 1)
                        bar = '█' * filled_len + '-' * (bar_len - filled_len)
                        print(f'\r[Archiving Progress] |{bar}| {percents}% ({self._progress_count}/{self._progress_total})', end='')

                    reverse_index += 1

                    # 如果反向展示完毕，重置标志
                    if reverse_index == len(self.images_to_show):
                        self.reverse_show_images = False  # 重置为正常播放模式
                        current_img = self.images_to_show[0]
                        if hasattr(self, 'video_writer'):
                            self.video_writer.release()
                            print(f"\n[INFO] Video Successfully Saved to: {next_video_path}")
                            del self.video_writer
                        self.images_to_show.clear()
                        reverse_index = 0
                        # 清理进度条变量
                        if hasattr(self, '_progress_total'):
                            del self._progress_total
                        if hasattr(self, '_progress_count'):
                            del self._progress_count
                        self.archiving = False

                elif self.images_to_show:
                    current_img = self.images_to_show[-1]
                    if self.reverse_show_images == True:
                        self.reverse_show_images = False

                # 获取当前屏幕分辨率
                screen_width = cv2.getWindowImageRect('Image Display')[2]
                screen_height = cv2.getWindowImageRect('Image Display')[3]

                if current_img is not None:
                    h1, w1 = current_img.shape[:2]
                    h2, w2 = frame.shape[:2]

                    # 计算图片显示区域的宽度比例
                    ratio1 = min(screen_width // 2 / w1, screen_height / h1)
                    ratio2 = min(screen_width // 2 / w2, screen_height / h2)

                    # 调整图片大小
                    new_w1, new_h1 = int(w1 * ratio1), int(h1 * ratio1)
                    new_w2, new_h2 = int(w2 * ratio2), int(h2 * ratio2)

                    resized_img1 = cv2.resize(current_img, (new_w1, new_h1))
                    resized_img2 = cv2.resize(frame, (new_w2, new_h2))

                    # 创建黑色背景
                    combined_image = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)

                    # 放置两张图片在黑色背景上
                    combined_image[(screen_height - new_h1) // 2:(screen_height - new_h1) // 2 + new_h1, 
                                (screen_width // 2 - new_w1) // 2:(screen_width // 2 - new_w1) // 2 + new_w1] = resized_img1

                    combined_image[(screen_height - new_h2) // 2:(screen_height - new_h2) // 2 + new_h2, 
                                screen_width // 2 + (screen_width // 2 - new_w2) // 2:screen_width // 2 + (screen_width // 2 - new_w2) // 2 + new_w2] = resized_img2

                else:
                    h2, w2 = frame.shape[:2]
                    ratio2 = min(screen_width // 2 / w2, screen_height / h2)
                    new_w2, new_h2 = int(w2 * ratio2), int(h2 * ratio2)
                    resized_img2 = cv2.resize(frame, (new_w2, new_h2))

                    # 创建黑色背景
                    combined_image = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)

                    # 放置历史帧在右侧，左侧为空
                    combined_image[(screen_height - new_h2) // 2:(screen_height - new_h2) // 2 + new_h2, 
                                screen_width // 2 + (screen_width // 2 - new_w2) // 2:screen_width // 2 + (screen_width // 2 - new_w2) // 2 + new_w2] = resized_img2

                cv2.imshow('Image Display', combined_image)
                # 计算展示内容所用时间
                elapsed_time = time.time() - start_time
                wait_time = max(0, int(30 - elapsed_time * 1000))  # 计算需要等待的时间
                
                if wait_time > 0:
                    cv2.waitKey(wait_time)
                else:
                    cv2.waitKey(1)  # 不等待或最小等待时间1毫秒


            for frame in reversed(frames):
                start_time = time.time()
                # 判断是否需要反向展示
                if self.reverse_show_images and self.images_to_show:
                    if reverse_index == 0:
                        # 初始化 VideoWriter
                        video_files = [f for f in os.listdir(self.history_dir) if f.endswith('.mp4')]
                        next_video_idx = len(video_files)
                        next_video_path = os.path.join(self.history_dir, f"{next_video_idx}.mp4")
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        img_for_shape = self.images_to_show[-1]
                        height, width = img_for_shape.shape[:2]
                        print(f"[INFO] Start Saving Video to {next_video_path}")
                        self.video_writer = cv2.VideoWriter(next_video_path, fourcc, 10, (width, height))
                        self._progress_total = len(self.images_to_show)
                        self._progress_count = 0

                    reverse_index = reverse_index % len(self.images_to_show)
                    current_img = self.images_to_show[-reverse_index - 1]  # 从最后一帧开始反向播放

                    # 写入当前帧到视频
                    if hasattr(self, 'video_writer') and current_img is not None:
                        self.video_writer.write(current_img)
                        self._progress_count += 1
                        # 打印进度条
                        bar_len = 30
                        filled_len = int(round(bar_len * self._progress_count / float(self._progress_total)))
                        percents = round(100.0 * self._progress_count / float(self._progress_total), 1)
                        bar = '█' * filled_len + '-' * (bar_len - filled_len)
                        print(f'\r[Archiving Progress] |{bar}| {percents}% ({self._progress_count}/{self._progress_total})', end='')

                    reverse_index += 1

                    # 如果反向展示完毕，重置标志
                    if reverse_index == len(self.images_to_show):
                        self.reverse_show_images = False  # 重置为正常播放模式
                        current_img = self.images_to_show[0]
                        if hasattr(self, 'video_writer'):
                            self.video_writer.release()
                            print(f"\n[INFO] Video Successfully Saved to: {next_video_path}")
                            del self.video_writer
                        self.images_to_show.clear()
                        reverse_index = 0
                        # 清理进度条变量
                        if hasattr(self, '_progress_total'):
                            del self._progress_total
                        if hasattr(self, '_progress_count'):
                            del self._progress_count

                elif self.images_to_show:
                    current_img = self.images_to_show[-1]
                    if self.reverse_show_images == True:
                        self.reverse_show_images = False

                # 获取当前屏幕分辨率
                screen_width = cv2.getWindowImageRect('Image Display')[2]
                screen_height = cv2.getWindowImageRect('Image Display')[3]

                if current_img is not None:
                    h1, w1 = current_img.shape[:2]
                    h2, w2 = frame.shape[:2]

                    # 计算图片显示区域的宽度比例
                    ratio1 = min(screen_width // 2 / w1, screen_height / h1)
                    ratio2 = min(screen_width // 2 / w2, screen_height / h2)

                    # 调整图片大小
                    new_w1, new_h1 = int(w1 * ratio1), int(h1 * ratio1)
                    new_w2, new_h2 = int(w2 * ratio2), int(h2 * ratio2)

                    resized_img1 = cv2.resize(current_img, (new_w1, new_h1))
                    resized_img2 = cv2.resize(frame, (new_w2, new_h2))

                    # 创建黑色背景
                    combined_image = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)

                    # 放置两张图片在黑色背景上
                    combined_image[(screen_height - new_h1) // 2:(screen_height - new_h1) // 2 + new_h1, 
                                (screen_width // 2 - new_w1) // 2:(screen_width // 2 - new_w1) // 2 + new_w1] = resized_img1

                    combined_image[(screen_height - new_h2) // 2:(screen_height - new_h2) // 2 + new_h2, 
                                screen_width // 2 + (screen_width // 2 - new_w2) // 2:screen_width // 2 + (screen_width // 2 - new_w2) // 2 + new_w2] = resized_img2

                else:
                    h2, w2 = frame.shape[:2]
                    ratio2 = min(screen_width // 2 / w2, screen_height / h2)
                    new_w2, new_h2 = int(w2 * ratio2), int(h2 * ratio2)
                    resized_img2 = cv2.resize(frame, (new_w2, new_h2))

                    # 创建黑色背景
                    combined_image = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)

                    # 放置历史帧在右侧，左侧为空
                    combined_image[(screen_height - new_h2) // 2:(screen_height - new_h2) // 2 + new_h2, 
                                screen_width // 2 + (screen_width // 2 - new_w2) // 2:screen_width // 2 + (screen_width // 2 - new_w2) // 2 + new_w2] = resized_img2

                cv2.imshow('Image Display', combined_image)
                # 计算展示内容所用时间
                elapsed_time = time.time() - start_time
                wait_time = max(0, int(30 - elapsed_time * 1000))  # 计算需要等待的时间
                
                if wait_time > 0:
                    cv2.waitKey(wait_time)
                else:
                    cv2.waitKey(1)  # 不等待或最小等待时间1毫秒

            capture.release()

            video_index += 1


        cv2.destroyAllWindows()
     
    def process_and_send_images(self, conn):
        ground_img = Image.open("test/image/ground.jpg")
        scale = 0.5
        ground_img = ground_img.resize((int(ground_img.width * scale), int(ground_img.height * scale)))
        self.input_image = ground_img  # 将初始图像赋值给类实例变量
        send_image(self.input_image, conn)
        
        while True:
            if not self.reset_event.is_set():
                time.sleep(0.1)  
                continue
            
            if self.latest_mask is not None:
                self._process_latest_mask(conn)
            time.sleep(0.001)   

    def _process_latest_mask(self, conn):
        if self.input_image is None:
            print("input_image is None, skipping processing.")
            ground_img = Image.open("test/image/ground.jpg")
            scale = 0.5
            self.input_image = ground_img.resize((int(ground_img.width * scale), int(ground_img.height * scale)))
            return
        time0 = time.time()
        mask_img = self.latest_mask
        x, y = self.latest_mask_info
        self.latest_mask = None
        self.latest_mask_info = None

        prompt_idx = self.prompt_idx
        strength = 0.75
        guidance_scale = 1.5
        square_ground = self.input_image.crop((x, y, x + 512, y + 512))

        start_time = time.time()
        img2img_result = img2img(square_ground, Image.fromarray(mask_img), prompt_idx)

        start_time = time.time()
        images = generate_interpolated_frames(square_ground, img2img_result, exp=4)

        self.images_to_show.append(cv2.cvtColor(np.array(self.input_image), cv2.COLOR_RGB2BGR))
        start_time = time.time()
        final_image = None
        for interpolated_image in images:
            if not self.reset_event.is_set():
                time.sleep(0.1)  
                continue
            final_image = merge_images(self.input_image, interpolated_image, Image.fromarray(mask_img), x, y)
            self.images_to_send.append(final_image)
            self.images_to_show.append(cv2.cvtColor(np.array(final_image), cv2.COLOR_RGB2BGR))

        self.input_image = final_image

    def prompt_idx_server(self, host, port):
        # 监听来自Unity的prompt_idx
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((host, port))
            s.listen(1)
            print(f"Prompt server listening on {host}:{port}")

            conn, addr = s.accept()
            with conn:
                print(f"Connected by {addr}")
                while True:
                    data = conn.recv(1024)
                    if not data:
                        break
                    self.prompt_idx = int(data.decode('utf-8').strip())
                    if self.hmd_status == "HMD Mounted":
                        print(prompts_list[self.prompt_idx],"\n")

def main():
    server = Server(host='127.0.0.1', port=12345)

    hmd_status_thread = threading.Thread(target=server.hmd_status_listener, args=('127.0.0.1', 8080))
    hmd_status_thread.start()
    
    mask_thread = threading.Thread(target=server.generate_mask_image_periodically)
    mask_thread.start()

    server_thread = threading.Thread(target=server.start_server)
    server_thread.start()

    save_thread = threading.Thread(target=server.save_images_thread)  # 新增存储线程
    save_thread.start()

    show_thread = threading.Thread(target=server.show_images_thread)  # 新增展示线程
    show_thread.start()

    prompt_thread = threading.Thread(target=server.prompt_idx_server, args=('127.0.0.1', 13000))
    prompt_thread.start()

    # 启动按键监听线程
    keyboard_thread = threading.Thread(target=server.keyboard_listener)
    keyboard_thread.start()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((server.host, server.port))
        s.listen()
        print(f"Server listening on {server.host}:{server.port}")

        conn, addr = s.accept()
        with conn:
            print(f"Connected by {addr}")
            send_thread = threading.Thread(target=server.send_images_thread, args=(conn,))
            send_thread.start()

            server.process_and_send_images(conn)

        print("Server will close now.")

if __name__ == "__main__":
    main()
