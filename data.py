import torch, os
from utils import spacetime_block
import constants

num_files = 2
directory = "data/valid/"

size = 512
steps = 128

start = 0
while os.path.isfile(directory + str(start) + ".pt"):
    # foo = torch.load(directory + str(start) + ".pt").permute(1, 0, 2, 3)
    # torch.save(foo, directory + str(start) + ".pt")
    start += 1
# exit()
for i in range(start, start + num_files):
    universe = spacetime_block(steps, size, size)
    torch.save(universe, directory + str(i) + ".pt")