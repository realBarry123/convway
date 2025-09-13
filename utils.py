import torch
import torch.nn.functional as F

def upscale(tensor, factor):
    return F.interpolate(tensor, scale_factor=factor, mode='bilinear')

def downscale(tensor, factor):
    avg_pool = torch.nn.AvgPool2d(kernel_size=factor)
    return avg_pool(tensor)
