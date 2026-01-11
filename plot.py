import torch
from model import ConvwayNet

import matplotlib.pyplot as plt
from matplotlib import animation
from constants import *

LOAD_PATH = f"models/{input("Enter model name: ")}.pt"
state_dict, configs, epoch = torch.load(LOAD_PATH)
model = ConvwayNet(**configs).to(DEVICE)
model.load_state_dict(state_dict)
model.eval()

fig, ax = plt.subplots()
state = torch.load(f"data/train/0.pt")[:,:1].to(DEVICE)
im = ax.imshow(state[0][0].clone().detach().to("cpu").numpy(), vmin=0, vmax=1, cmap="viridis")
n_frames = 10
frames = [state,]

for i in range(n_frames):
    with torch.no_grad():
        print(state.shape)
        state, _ = model(state)
        frames.append(state)

def update(state):
    im.set_data(state[0][0].clone().detach().to("cpu").numpy())

game = animation.FuncAnimation(fig, update, frames=frames)
plt.show()
