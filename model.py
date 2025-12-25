from collections.abc import Callable
import torch
from torch import nn
from torchsummary import summary
from constants import *

B = 32

def get_conv_stack(channels: list[int], kernel_size, activation=None) -> list[Callable]: 
    layers = []
    for i in range(len(channels) - 1): # each pair of channel sizes
        layers.append(nn.Conv2d(
            in_channels=channels[i], 
            out_channels=channels[i+1], 
            kernel_size=kernel_size,
            padding="same"
        ))
        if activation == "relu": layers.append(nn.LeakyReLU())
        elif activation == "sin": layers.append(torch.sin)

    return layers


class ConvwayNet(nn.Module): 

    def __init__(self, conv_channels: list[int], activation):
        super().__init__()

        assert conv_channels[0] == 1
        assert conv_channels[-1] == 1

        self.configs = {
            "conv_channels": conv_channels,
            "activation": activation
        }

        self.convs = nn.Sequential(*get_conv_stack(conv_channels, kernel_size=SCALE*3, activation=activation))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor):  # (B, C=1, H, W)

        r = self.convs(x)
        x = self.sigmoid(x + r)
        
        return x


if __name__ == "__main__":
    model = ConvwayNet(conv_channels=[1, 4, 1], do_relu=True)
    summary(model, input_size=(1, 69, 420), batch_size=B)