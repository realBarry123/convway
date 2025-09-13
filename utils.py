import torch
import torch.nn.functional as F

# Upscale spatial dimensions in a blurry way
def upscale(tensor, factor):
    return F.interpolate(tensor, scale_factor=factor, mode='bilinear')

# Downscale spatial dimensions by mean pool
def downscale(tensor, factor):
    avg_pool = torch.nn.AvgPool2d(kernel_size=factor)
    return avg_pool(tensor)
