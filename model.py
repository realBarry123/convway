from collections.abc import Callable
import torch
from torch import nn
from constants import *

B = 32

def get_conv_stack(channels: list[int], kernel_size=SCALE*3, do_relu=True) -> list[Callable]:
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

def get_smusher(mode: str):
    VALID_MODES = ["last", "4th", "mean", "conv"]
    if mode not in VALID_MODES:
            raise ValueError(f"invalid smushing mode: expected one of [\"{"\", \"".join(VALID_MODES)}\"] but got \"{squeeze_mode}\"")
    if mode == "last":
        return lambda x: x[:, SCALE-1:, :, :]
    if mode == "4th":
        return lambda x: x[:, :1, :, :]
    if mode == "mean":
        return lambda x: torch.mean(x, dim=1, keepdim=True)
    if mode == "conv": 
        return torch.nn.Conv2d(
            in_channels=4, 
            out_channels=1, 
            kernel_size=(SCALE * 3, SCALE * 3), 
            padding="same"
        )


class ConvwayNet(torch.nn.Module):

    def __init__(self, x_smush_mode: str, r_smush_mode: str, conv_channels: list[int], do_relu: bool):
        super().__init__()

        self.configs = {
            "x_smush_mode": x_smush_mode,
            "r_smush_mode": r_smush_mode, 
            "conv_channels": conv_channels,
            "do_relu": do_relu
        }
        
        # Downscaling (T) layer 
        self.smush_x = get_smusher(x_smush_mode)
        self.smush_r = get_smusher(r_smush_mode)

        # Conv layers
        self.convs = nn.Sequential(*get_conv_stack(conv_channels, do_relu=do_relu))

        # Activations
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor):  # (B, T=4, H, W)
        r = self.smush_r(x)  # (B, T=1, H, W)

        r = self.convs(r)  # (B, T=1, H, W)

        x = self.smush_x(x)
        x = self.sigmoid(x + r)
        
        return x

# Test code does not run on import
if __name__ == "__main__":
    model = ConvwayNet(squeeze_mode="n", conv_channels=[1, 4, 1], do_relu=True)
    print(model(torch.zeros((B, SCALE, 69, 420))).shape)