import torch, pygame, numpy

pygame.init()
screen = pygame.display.set_mode((1280, 720))
clock = pygame.time.Clock()
running = True
paused = True

def show_matrix(matrix, cell_size=10):
    for row in range(len(matrix)):
        for col in range(len(matrix[row])):
            left = col * cell_size
            top = row * cell_size
            value = round(matrix[row][col] * 255)  # the colour of the cell
            pygame.draw.rect(screen, (value, value, value), (left, top, cell_size, cell_size))

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:  # user closed window
            running = False

    screen.fill("black")

    # RENDER YOUR GAME HERE

    show_matrix(numpy.array([
        [0, 0, 0, 0.8],
        [0, 1, 0.5, 0],
        [1, 0.7, 1, 1],
        [0.2, 0.3, 0.4, 0.5]
    ]))

    # flip() the display to put your work on screen
    pygame.display.flip()

    clock.tick(60)  # limits FPS to 60

pygame.quit()