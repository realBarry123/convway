import torch, pygame, numpy
import utils

def draw_matrix(matrix, screen, cell_size=10):
    for row in range(len(matrix)):
        for col in range(len(matrix[row])):
            left = col * cell_size
            top = row * cell_size
            value = round(matrix[row][col] * 255)  # the colour of the cell
            pygame.draw.rect(screen, (value, value, value), (left, top, cell_size, cell_size))

def display_matrix(matrix: torch.Tensor):
    pygame.init()
    screen = pygame.display.set_mode((1280, 720))
    clock = pygame.time.Clock()
    running = True
    paused = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:  # user closed window
                running = False

        screen.fill("black")

        # RENDER YOUR GAME HERE

        draw_matrix(matrix.tolist(), screen)

        # flip() the display to put your work on screen
        pygame.display.flip()

        clock.tick(60)  # limits FPS to 60

    pygame.quit()



