import torch, random
import torch.nn.functional as F
from tqdm import tqdm

from lifegame import update_game

# Upscale spatial dimensions in a blurry way
def upscale(tensor, time_factor, space_factor):
    tensor = tensor.unsqueeze(1)
    tensor = F.interpolate(tensor, scale_factor=(time_factor, space_factor, space_factor), mode='trilinear')
    tensor = tensor.squeeze(1)
    return tensor

# Downscale spatial dimensions by mean pool
def downscale(tensor, factor):
    avg_pool = torch.nn.AvgPool2d(kernel_size=factor)
    return avg_pool(tensor)

def trimmed_spacetime_block(steps, factor, height, width, batch_size=1):
    out = spacetime_block(steps, factor, height, width, batch_size, time_factor=factor+1)
    mask = torch.arange(out.shape[1]) % factor != 0
    out = out[:, mask]
    return out

def spacetime_block(steps, factor, height, width, batch_size=1, time_factor=None):
    # assert batch_size == 1, "batch size not implemented for any number other than 1"
    if height % factor != 0 or width % factor != 0:
        raise ValueError("height and width dimensions must be divisible by factor")
    if time_factor == None: 
        time_factor = factor
    probability = random.triangular(0, 0.6, 0.3)

    # Generate initial state
    states = torch.bernoulli(
        input=torch.full(
            size=(batch_size, 1, int(height/4), int(width/4)),
            fill_value=probability
        )
    )
    # (B, 1, H/4, W/4)
    
    for t in tqdm(range(steps), desc=f"Generating Data"): 
        new_state = torch.empty(batch_size, 1, int(height/4), int(width/4))
        for i in range(batch_size): 
            new_state[i][0] = update_game(states[i][t])
        states = torch.cat((states, new_state), dim=1)

    states = upscale(states, time_factor, factor)
    # states = states.permute(1, 0, 2, 3) # spacetime block (B=1, (steps+T+1) * 4 , H, W)
    return states

# print(spacetime_block(8, 4, 128, 128, batch_size=2).shape)