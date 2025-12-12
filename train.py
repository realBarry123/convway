import torch, random, time, wandb
from tqdm import tqdm
from model import ConvwayNet
from demo import display_matrix
import utils
from constants import *

B = 1
assert B == 1, "batch size bigger than 1 has not been implemented yet"
H = 512  # training height
W = 512  # training width

SAVE_PATH = "models/te.pt"

NUM_EPOCHS = 1
SIM_STEPS = 32  # how many steps to simulate per epoch
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

try: 
    state_dict, configs, epoch = torch.load(SAVE_PATH)
    model = ConvwayNet(**configs).to(DEVICE)
    model.load_state_dict(state_dict)
    start_epoch = epoch + 1
except FileNotFoundError:
    model = ConvwayNet(x_smush_mode="last", r_smush_mode="mean", conv_channels=(1, 4, 1), do_relu=True).to(DEVICE)

optimizer = torch.optim.Adam(params=model.parameters(), lr=LR, weight_decay=0.01)
mse_loss = torch.nn.MSELoss()


def save(model: torch.nn.Module, epoch: int, save_path: str):
    torch.save([model.state_dict(), model.configs, epoch], save_path)


def train_epoch(model: torch.nn.Module, epoch: int = 0):
    model.train()

    total_loss = 0

    states = torch.load(f"data/train/{random.randint(0, 7)}.pt")

    for step in tqdm(range(SCALE * SIM_STEPS), desc=f"E{epoch} Train"):
        x = states[:, step: step + SCALE]
        target = states[:, step + SCALE: step + SCALE + 1]
        y = model(x)
        loss = mse_loss(y, target)
        
        if DO_WANDB: 
            wandb_run.log({"loss": loss.item()})

        total_loss += loss.item()
        
        # The holy trinity
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f"Train loss (average): {total_loss / (SCALE * SIM_STEPS)}")


def valid_epoch(model: torch.nn.Module, epoch: int = 0) -> int:
    model.eval()
    with torch.no_grad():
        total_loss = 0
        states = torch.load("data/valid/0.pt")
        for step in tqdm(range(SCALE * SIM_STEPS), desc=f"E{epoch} Valid"):

            x = states[:, step: step + SCALE]
            target = states[:, step + SCALE: step + SCALE + 1]
            y = model(x)
            loss = mse_loss(y, target)

            total_loss += loss.item()
        print(f"Valid loss (average): {total_loss / (SCALE * SIM_STEPS)}")


for epoch in range(start_epoch, start_epoch + NUM_EPOCHS):
    train_epoch(model, epoch)
    valid_epoch(model, epoch)
    save(model, epoch, SAVE_PATH)