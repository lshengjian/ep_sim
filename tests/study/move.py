import pygame

pygame.init()
width, height = 800, 600
window = pygame.display.set_mode((width, height))
pygame.display.set_caption("WSAD Movement")

target_x = width // 2
target_y = height // 2
movement_speed = 5

clock = pygame.time.Clock()

moving_direction = None

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_w:
                moving_direction = "up"
            elif event.key == pygame.K_s:
                moving_direction = "down"
            elif event.key == pygame.K_a:
                moving_direction = "left"
            elif event.key == pygame.K_d:
                moving_direction = "right"
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_w and moving_direction == "up":
                moving_direction = None
            elif event.key == pygame.K_s and moving_direction == "down":
                moving_direction = None
            elif event.key == pygame.K_a and moving_direction == "left":
                moving_direction = None
            elif event.key == pygame.K_d and moving_direction == "right":
                moving_direction = None

    if moving_direction == "up":
        target_y -= movement_speed
    elif moving_direction == "down":
        target_y += movement_speed
    elif moving_direction == "left":
        target_x -= movement_speed
    elif moving_direction == "right":
        target_x += movement_speed

    window.fill((0, 0, 0))
    pygame.draw.circle(window, (255, 255, 255), (target_x, target_y), 50)
    pygame.display.flip()

    clock.tick(60)

pygame.quit()