import torch, pygame, numpy
from model import ConvwayNet
import utils

LOAD_PATH = "models/model_2.pt"

def draw_matrix(matrix, screen, cell_size=10):
    for row in range(len(matrix)):
        for col in range(len(matrix[row])):
            left = col * cell_size
            top = row * cell_size
            value = round(matrix[row][col] * 255)  # the colour of the cell
            value = min(max(value, 0), 255)
            pygame.draw.rect(screen, (value, value, value), (left, top, cell_size, cell_size))

def display_matrix(matrix: torch.Tensor, cell_size=10):
    pygame.init()
    screen = pygame.display.set_mode((1280, 720))
    clock = pygame.time.Clock()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:  # user closed window
                running = False

        screen.fill("black")
        draw_matrix(matrix.tolist(), screen, cell_size)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

def play_game(model, cell_size=10):
    pygame.init()
    screen = pygame.display.set_mode((1280, 720))
    clock = pygame.time.Clock()
    running = True
    paused = True

    seed_state = torch.randint(0, 2, (1, 1, int(512/4), int(512/4))).float() #  (B=1, T=1, H, W)
    state = utils.upscale(seed_state, 4).repeat(1, 4, 1, 1)  # (B=1, T=4, H, W)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:  # user closed window
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:  # user pressed SPACE
                    paused = not paused
                elif event.key == pygame.K_RIGHT:
                    y_pred = model(state)
                    state = torch.cat((state, y_pred), dim=1)[:, -4:, :, :].detach()

        screen.fill("black")

        draw_matrix(state[-1][0].tolist(), screen, cell_size)

        if not paused:
            y_pred = model(state)
            state = torch.cat((state, y_pred), dim=1)[:, -4:, :, :].detach()
        
        pygame.display.flip()
        clock.tick(10)

    pygame.quit()

if __name__ == "__main__":
    model = ConvwayNet()
    state_dict, epoch = torch.load(LOAD_PATH)
    model.load_state_dict(state_dict)
    model.eval()

    play_game(model, cell_size=3)