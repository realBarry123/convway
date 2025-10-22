import torch, random, time, wandb
from tqdm import tqdm
from model import ConvwayNet
from demo import display_matrix
import utils

B = 1
assert B == 1, "batch size bigger than 1 has not been implemented yet"
T = 4
CHAIN_DEPTH = 2
H = 1024  # training height
W = 1024  # training width
T_SMOOTH_WEIGHT = 0.1

SAVE_PATH = "models/interpolate_32_steps_4.pt"

NUM_EPOCHS = 4
SIM_STEPS = 32  # how many steps to simulate per epoch
LR = 0.001
DEVICE = "cpu"

SAVING = True
DO_WANDB = False

if DO_WANDB: 
    wandb_run = wandb.init(
        project="convway",
        entity="barry-and-only-barry",
        config={
            "lr": 0.001,
            "epochs": 10,
        },
        id="lilac-rain-2",
        resume=True
    )


start_epoch = 0

model = ConvwayNet().to(DEVICE)

if SAVING:
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

    # BUILD SPACETIME BLOCK
    states = utils.spacetime_block(steps=SIM_STEPS, factor=T, height=H, width=W, batch_size=B)
    print(f"Created spacetime block: {states.shape}")
    
    EPOCH_SIZE = states.shape[0] - (T + 1) + 1
    print(f"Training for {EPOCH_SIZE} steps...")
    # exit()

    for step in tqdm(range(EPOCH_SIZE), desc=f"E{epoch} Train"):
        x = states[step:step+T].permute(1, 0, 2, 3)
        target = states[step+T]
        y = model(x).squeeze(1)
        loss = mse_loss(y, target)
        
        if DO_WANDB: 
            wandb_run.log({"loss": loss.item()})

        total_loss += loss.item()
        
        # The holy trinity
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f"Train loss (average): {total_loss/EPOCH_SIZE}")
    
    if SAVING:
        torch.save([model.state_dict(), epoch], SAVE_PATH)
