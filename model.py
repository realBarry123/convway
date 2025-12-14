from collections.abc import Callable
import torch
from torch import nn
from torchsummary import summary
from constants import *

B = 32

def get_conv_stack(channels: list[int], kernel_size, do_relu=True) -> list[Callable]: 
    layers = []
    for i in range(len(channels) - 1):
        layers.append(nn.Conv2d(
            in_channels=channels[i], 
            out_channels=channels[i+1], 
            kernel_size=kernel_size,
            padding="same"
        ))
        if do_relu: layers.append(nn.LeakyReLU())

    return layers

# TODO: force symmetrical conv kernels
class ConvwayNet(torch.nn.Module): 

    def __init__(self, conv_channels: list[int], do_relu):
        super().__init__()

        assert conv_channels[0] == 1
        assert conv_channels[-1] == 1

        self.configs = {
            "conv_channels": conv_channels,
            "do_relu": do_relu
        }

        # Conv layers
        self.convs = nn.Sequential(*get_conv_stack(conv_channels, kernel_size=SCALE*3, do_relu=do_relu))

        # Activations
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor):  # (B, C=1, H, W)

        r = self.convs(x)
        x = self.sigmoid(x + r)
        
        return x

# Test code does not run on import
if __name__ == "__main__":
    model = ConvwayNet(conv_channels=[1, 4, 1], do_relu=True)
    summary(model, input_size=(1, 69, 420), batch_size=B)