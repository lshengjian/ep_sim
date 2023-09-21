import pygame
from PIL import Image

# 初始化 Pygame
pygame.init()

# 创建窗口
width, height = 800, 600
screen = pygame.display.set_mode((width, height))

# 创建 Surface
surface = pygame.Surface((200, 200))
surface.fill((255, 0, 0))  # 填充红色

# 将 Surface 转换为图像
image = pygame.image.tostring(surface, 'RGB')
pil_image = Image.frombytes('RGB', (surface.get_width(), surface.get_height()), image)

# 打印 Surface 和图像的高度
print("Surface 高度:", surface.get_height())
print("图像高度:", pil_image.height)

# 退出 Pygame
pygame.quit()