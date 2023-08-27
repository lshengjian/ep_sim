import pygame
import numpy as np
from epsim.renderers.rendering import set_color,rotate_fn
from epsim.renderers.shapes import get_shape
# Initialize Pygame
pygame.init()

# Create a window
window_width = 800
window_height = 600
window = pygame.display.set_mode((window_width, window_height))

# Set the background color
background_color = (0, 0, 0)
window.fill(background_color)

img1=get_shape(1)

img2=get_shape(11)
img2=set_color(img2,255,0,0)
window.blit(pygame.surfarray.make_surface(img1),(10,10))
window.blit(pygame.surfarray.make_surface(img2),(100,100))

img3=get_shape(1)
assert (img1==img3).all()
pygame.display.flip()

# Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

# Quit Pygame
pygame.quit()
