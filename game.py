import pygame
import numpy as np
from epsim.renderers.rendering import set_color,blend_imgs
from epsim.renderers.shapes import get_slot_shape,get_crane_shape,get_workpiece_shape
# Initialize Pygame

def make_surface(img):
    img= np.transpose(img, axes=(1, 0, 2))
    return pygame.surfarray.make_surface(img)


pygame.init()

# Create a window
window_width = 800
window_height = 600
window = pygame.display.set_mode((window_width, window_height))

# Set the background color
background_color = (0, 0, 0)
window.fill(background_color)

img1=get_slot_shape(1)

img2=get_workpiece_shape('A')
img2=set_color(img2,0,255,0)
img=blend_imgs(img2,img1,(0,0))


window.blit(make_surface(img1),(10,10))
window.blit(make_surface(img2),(100,10))


window.blit(make_surface(img),(50,200))

# img3=get_shape(1)
# assert (img1==img3).all()
pygame.display.flip()

# Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

# Quit Pygame
pygame.quit()
