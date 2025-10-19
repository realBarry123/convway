import torch, random
import torch.nn.functional as F

import utils
from lifegame import update_game

# Upscale spatial dimensions in a blurry way
def upscale(tensor, factor):
    tensor = tensor.unsqueeze(1)
    tensor = F.interpolate(tensor, scale_factor=factor, mode='trilinear')
    tensor = tensor.squeeze(1)
    return tensor

# Downscale spatial dimensions by mean pool
def downscale(tensor, factor):
    avg_pool = torch.nn.AvgPool2d(kernel_size=factor)
    return avg_pool(tensor)

def spacetime_block(steps, height, width, batch_size=1):
    assert batch_size == 1, "batch size not implemented for any number other than 1"
    assert height % 4 == 0 and width % 4 == 0, "dimensions must be multiple of 4"
    probability = random.triangular(0, 1, 0.4)
    states = torch.bernoulli(
        input=torch.full(
            size=(batch_size, 1, int(height/4), int(width/4)),
            fill_value=probability
        )
    )
    
    for i in range(steps): 
        new_state = update_game(states[0][i])
        new_state = new_state.unsqueeze(0).unsqueeze(0) # (B=1, 1, H/4, W/4)
        states = torch.cat((states, new_state), dim=1)

    states = utils.upscale(states, 4)
    states = states.permute(1, 0, 2, 3) # spacetime block (B=1, (steps+T+1) * 4 , H, W)
    return states