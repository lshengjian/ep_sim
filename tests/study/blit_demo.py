import pygame
import numpy as np

# 初始化 Pygame
pygame.init()

# 创建窗口
width, height = 800, 600
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Blit Example")

# 生成图像数据
image_size = (200, 200)
image_array = np.random.randint(0, 256, size=(image_size[1], image_size[0], 3), dtype=np.uint8)
image_surface = pygame.surfarray.make_surface(image_array)

# 主循环
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # 清空屏幕
    screen.fill((0, 0, 0))

    # 将图像数据绘制到屏幕上
    screen.blit(image_surface, (width // 2 - image_size[0] // 2, height // 2 - image_size[1] // 2))

    # 更新屏幕显示
    pygame.display.flip()

# 退出 Pygame
pygame.quit()