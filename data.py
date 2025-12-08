import torch, os
from utils import spacetime_block
import constants

num_files = 5
size = 1024
steps = 32

start = 0
while os.path.isfile("data/universe" + str(start)):
    start += 1

for i in range(start, start + num_files):
    universe = spacetime_block(steps, constants.SCALE, size, size)
    torch.save(universe, "data/universe" + str(i))