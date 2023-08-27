import pygame
import numpy as np
import time
# Initialize Pygame
pygame.init()

# Create a window
window_width = 800
window_height = 600

IMG_SIZE=np.array([32,64],dtype=np.float32) #x,y
window = pygame.display.set_mode((window_width, window_height))

# Set the background color
background_color = (0, 0, 0)
window.fill(background_color)

# Define the number of sides of the polygon
num_sides = 6  # Change this to the desired number of sides

# Calculate the angle between each side of the polygon
angle = 2 * np.pi / num_sides

# Set the radius of the polygon
radius = 200

# Calculate the coordinates of the center of the window
center_x = window_width // 2
center_y = window_height // 2

# List to store the vertices of the polygon
vertices = []

# Set the color of the polygon
polygon_color = (255, 255, 255)

# Calculate the coordinates of each vertex of the polygon
for i in range(num_sides):
    x = np.cos(i * angle)+0.5
    y = np.sin(i * angle)+0.5
    vertices.append(np.array([x, y]))

# Create a surface for the polygon
surface = pygame.Surface(IMG_SIZE, pygame.SRCALPHA) #
ds=[]
for v in vertices:
    x,y=v*IMG_SIZE
    ds.append([int(x+0.5),int(y+0.5)]) 
# Draw the polygon on the surface
pygame.draw.polygon(surface, polygon_color, ds)
data=np.array(pygame.surfarray.pixels3d(surface))
# data=np.transpose(
#                 np.array(pygame.surfarray.pixels3d(surface)), axes=(1, 0, 2)
#             )
print(data.shape)
window.blit(surface,(10,10))
s2=pygame.surfarray.make_surface(data)
window.blit(s2,(100,100))
pygame.display.flip()
# Convert the surface to a NumPy array
# array = np.frombuffer(surface.get_buffer(), dtype=np.uint8)
# # Reshape the array to match the shape of the surface
# array = array.reshape((window_height, window_width, 4))


# Print the shape of the array

time.sleep(3)
# Quit Pygame
pygame.quit()
