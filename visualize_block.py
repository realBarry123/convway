import pygame

from utils import spacetime_block, trimmed_spacetime_block
from demo import draw_matrix

pygame.init()
screen = pygame.display.set_mode((1280, 720))
clock = pygame.time.Clock()
running = True
paused = True
i = 0

SIM_STEPS = 8

block = spacetime_block(steps=SIM_STEPS, factor=4, height=128, width=128, batch_size=1)

pygame.font.init()
my_font = pygame.font.SysFont('Comic Sans MS', 30)

while running and i <= SIM_STEPS * 4:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:  # user closed window
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:  # user pressed SPACE
                paused = not paused
            elif event.key == pygame.K_RIGHT:
                i += 1

    screen.fill("black")
    draw_matrix(block[0][i].tolist(), screen, 5)

    text_surface = my_font.render(str(i), False, (255, 255, 255))
    screen.blit(text_surface, (0, 0))

    pygame.display.flip()
    if not paused:
        i = i+1
    clock.tick(20)

pygame.quit()