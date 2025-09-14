import torch
from tqdm import tqdm
from model import ConvwayNet
from lifegame import update_game
from demo import display_matrix
import utils

B = 1
T = 4
H = 1024  # training height
W = 1024  # training width

SAVE_PATH = "models/model_1.pt"

NUM_EPOCHS = 10
EPOCH_SIZE = 8  # size of epoch
LR = 0.001
DEVICE = "cpu"

start_epoch = 0

model = ConvwayNet().to(DEVICE)

try: 
    state_dict, epoch = torch.load(SAVE_PATH)
    model.load_state_dict(state_dict)
    start_epoch = epoch + 1
except FileNotFoundError:
    pass

optimizer = torch.optim.Adam(params=model.parameters(), lr=LR, weight_decay=0.01)
mse_loss = torch.nn.MSELoss()

for epoch in range(start_epoch, start_epoch + NUM_EPOCHS):
    model.train()

    total_loss = 0

    seed_state = torch.randint(0, 2, (B, 1, int(H/4), int(W/4))).float()
    state = utils.upscale(seed_state, 4).repeat(1, T, 1, 1)

    for step in tqdm(range(EPOCH_SIZE), desc=f"E{epoch} Train"):
        # Downscale x
        x = state.clone()
        x_smushed = utils.downscale(torch.mean(x, dim=1, keepdim=False), T)  # (B, 1, H/4, W/4)
        x_binary = torch.heaviside(x_smushed - 0.5, values=torch.tensor([0.]))  # (B, H/4, W/4)

        # Conversion
        y_binary = torch.stack([update_game(universe) for universe in x_binary], dim=0)

        # Upscale y
        y_binary = torch.unsqueeze(y_binary, dim=1)  # (B, 1, H/4, W/4)
        y = utils.upscale(y_binary, T)  # (B, 1, H, W)
        
        # Calculate loss
        y_pred = model(x)
        loss = mse_loss(y, y_pred) + 0.1 * mse_loss(y[0][0], state[0][-1])
        total_loss += loss.item()
        
        # The holy trinity
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Add to and trim state
        state = torch.cat((state, y_pred), dim=1)[:, -4:, :, :].detach()
    
    print(f"Train loss (average): {total_loss/EPOCH_SIZE}")
    
    torch.save([model.state_dict(), epoch], SAVE_PATH)
