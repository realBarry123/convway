import torch, random, time
from tqdm import tqdm
from model import ConvwayNet
from lifegame import update_game
from demo import display_matrix
import utils

B = 1
T = 4
CHAIN_DEPTH = 3
H = 1024  # training height
W = 1024  # training width
BETA = 0.1

SAVE_PATH = "models/model_ord2.pt"

NUM_EPOCHS = 64
EPOCH_SIZE = 1  # size of epoch
LR = 0.0001
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

best_loss = 1
patience = 3

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

        for frame in range(CHAIN_DEPTH):
            # Conversion
            x_binary = torch.stack([update_game(universe) for universe in x_binary], dim=0)
        
        # Upscale y
        y_binary = torch.unsqueeze(x_binary, dim=1)  # (B, 1, H/4, W/4)
        y = utils.upscale(y_binary, T)  # (B, 1, H, W)

        smoothness_loss = torch.tensor(0.0, device=x.device)
        
        for frame in range(T * CHAIN_DEPTH):
            # Forward
            x = model(state)

            smoothness_loss += mse_loss(x, state[:, -1:])

            # Add to and trim state
            state = torch.cat((state, x), dim=1)[:, -4:, :, :].detach()

        loss = mse_loss(x, y) + BETA * smoothness_loss / CHAIN_DEPTH
        total_loss += loss.item()
        
        # The holy trinity
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    avg_loss = total_loss/EPOCH_SIZE

    print(f"Train loss (average): {avg_loss}")
    if avg_loss > best_loss:
        patience -= 1
    else: 
        best_loss = avg_loss
        if patience < 3: 
            patience += 1

    if patience == 0: 
        print("Stopping train loop...")
        break
    
    torch.save([model.state_dict(), epoch], SAVE_PATH)
