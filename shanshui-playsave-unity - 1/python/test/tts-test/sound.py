import os
import random
import pygame

# 初始化pygame的音频模块
pygame.mixer.init()

# 音频文件目录
audio_dir = r"C:\shanshui\python\tts"

# 获取目录中的所有MP3文件
audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.mp3')]

# 随机选择一个MP3文件
random_file = random.choice(audio_files)

# 构建完整的文件路径
audio_path = os.path.join(audio_dir, random_file)

# 加载并播放音频
pygame.mixer.music.load(audio_path)
pygame.mixer.music.play()

# 等待音频播放结束
while pygame.mixer.music.get_busy():
    pygame.time.Clock().tick(10)
