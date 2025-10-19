import torch, random, time, wandb
from tqdm import tqdm
from model import ConvwayNet
from lifegame import update_game
from demo import display_matrix
import utils

B = 1
assert B == 1, "batch size bigger than 1 has not been implemented yet"
T = 4
CHAIN_DEPTH = 2
H = 1024  # training height
W = 1024  # training width
T_SMOOTH_WEIGHT = 0.1

SAVE_PATH = "models/model_2.pt"

NUM_EPOCHS = 8
SIM_STEPS = 1  # how many steps you want to simulate
LR = 0.001
DEVICE = "cpu"

SAVING = False
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
    probability = random.random()
    states = torch.bernoulli(
        input=torch.full(
            size=(B, 1, int(H/4), int(W/4)),
            fill_value=probability
        )
    )
    
    for i in range(SIM_STEPS): 
        new_state = update_game(states[0][i])
        new_state = new_state.unsqueeze(0).unsqueeze(0) # (B=1, 1, H/4, W/4)
        states = torch.cat((states, new_state), dim=1)

    states = states.unsqueeze(1)
    states = utils.upscale(states, 4)
    states = states.squeeze(1)
    states = states.permute(1, 0, 2, 3) # spacetime block (B=1, (SIM_STEPS+T+1) * 4 , H, W)
    print(states.shape)
    
    EPOCH_SIZE = states.shape[0] - 1
    print(f"Training for {EPOCH_SIZE} epochs...")
    exit()

    for step in tqdm(range(EPOCH_SIZE), desc=f"E{epoch} Train"):
        
        if DO_WANDB: 
            wandb_run.log({"loss": loss.item()})
        
        # The holy trinity
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f"Train loss (average): {total_loss/EPOCH_SIZE}")
    
    if SAVING:
        torch.save([model.state_dict(), epoch], SAVE_PATH)
