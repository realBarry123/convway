import pygame

from utils import spacetime_block
from demo import draw_matrix

pygame.init()
screen = pygame.display.set_mode((1280, 720))
clock = pygame.time.Clock()
running = True
i = 0

SIM_STEPS = 32

block = spacetime_block(steps=SIM_STEPS, factor=4, height=512, width=512, batch_size=1)

pygame.font.init()
my_font = pygame.font.SysFont('Comic Sans MS', 30)

while running and i <= SIM_STEPS * 4:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:  # user closed window
            running = False

    screen.fill("black")
    draw_matrix(block[i][0].tolist(), screen, 5)

    text_surface = my_font.render(str(i/4), False, (255, 255, 255))
    screen.blit(text_surface, (0, 0))

    pygame.display.flip()
    i = i+1
    clock.tick(10)

pygame.quit()