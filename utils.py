import torch
import torch.nn.functional as F

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
