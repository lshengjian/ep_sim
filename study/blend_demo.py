import pygame
import numpy as np
from PIL import Image
import os

# 获取当前脚本的绝对路径
script_path = os.path.abspath(__file__)
print("脚本路径:", script_path)

# 获取当前脚本所在的目录
script_dir = os.path.dirname(script_path)
print("脚本所在目录:", script_dir)
# 初始化 Pygame
pygame.init()

# 创建窗口
width, height = 800, 600
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Sprite Blend Example")

# 载入背景图像
background_image = pygame.image.load(script_dir+"/alien2.png").convert_alpha()

# 生成随机形状的精灵
sprite_size = (64, 64)
sprite_array = np.zeros((sprite_size[1], sprite_size[0],3), dtype=np.uint8)
for x in range(sprite_size[1]):
    for y in range(sprite_size[0]):
        x1=x-sprite_size[1]//2
        y1=y-sprite_size[0]//2
        if x1**2+y1**2<32**2:
            sprite_array[x,y]=[255,0,0]
surface = pygame.surfarray.make_surface(sprite_array)
img=pygame.image.tostring(surface,'RGBA')
sprite=pygame.image.fromstring(img,(sprite_size[1], sprite_size[0]),'RGBA')
sprite.set_alpha(128)
#random.randint(0, 255, size=(sprite_size[1], sprite_size[0], 3), dtype=np.uint8)
#

# 将精灵转换为 PIL 图像对象，并保存为 PNG 格式
# sprite_image = Image.fromarray(sprite_array)
# sprite_image.save(script_dir+"/sprite.png")

# # 加载精灵图像
# sprite_image = pygame.image.load(script_dir+"/sprite.png").convert_alpha()

# 主循环
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # 清空屏幕
    screen.fill((0, 0, 0))
    background_image.blit(sprite, (0, 0))#, special_flags=pygame.BLEND_RGBA_MAX)

    # 将背景图像绘制到屏幕上
    screen.blit(background_image, (0, 0))


    # 更新屏幕显示
    pygame.display.flip()

# 退出 Pygame
pygame.quit()