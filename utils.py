import torch, random, math
import torch.nn.functional as F
from tqdm import tqdm

from constants import *
from lifegame import update_game

# Upscale spatial dimensions in a blurry way
def upscale(tensor, factor):
    tensor = F.interpolate(tensor, scale_factor=factor, mode='bilinear', align_corners=False)
    return tensor

# Downscale spatial dimensions by mean pool
def downscale(tensor, factor):
    avg_pool = torch.nn.AvgPool2d(kernel_size=factor)
    return avg_pool(tensor)

def trimmed_spacetime_block(steps, factor, height, width, batch_size=1):
    out = spacetime_block(steps, factor, height, width, batch_size, time_factor=factor+1)
    mask = torch.arange(out.shape[1]) % (factor + 1) != 0
    out = out[:, mask]
    return out

def spacetime_block(steps, height, width, batch_size=1):

    if height % SCALE != 0 or width % SCALE != 0:
        raise ValueError("height and width dimensions must be divisible by factor")
    
    probability = random.triangular(0.1, 0.6, 0.3)

    # Empty tensor and initial state
    states = torch.empty(batch_size, steps + 1, math.ceil(height/4), math.ceil(width/4))
    states[:, 0] = torch.bernoulli(
        input=torch.full(
            size=(batch_size, math.ceil(height/4), math.ceil(width/4)),
            fill_value=probability
        )
    )

    # Fill subsequent states
    for t in tqdm(range(steps), desc=f"Generating Data"): 
        for b in range(batch_size):
            states[b][t+1] = update_game(states[b][t])

    states = upscale(states, SCALE)

    if (states.shape[2] > height): 
        states = states[:, :, :height,:]
    elif (states.shape[3] > width):
        states = states[:, :, :height,:]

    return states

if __name__ == "__main__":
    test_block = spacetime_block(8, 128, 128, batch_size=2)
    print(test_block.shape)
    if torch.equal(test_block[0][0], test_block[0][1]):
        raise ValueError("duplicate timesteps")
        