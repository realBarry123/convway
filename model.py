import torch
from torch import nn

B = 32
T = 4

def create_conv_stack(channels, kernel_size=T*3, do_relu=True):
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


class ConvwayNet(torch.nn.Module):

    def __init__(self, squeeze_mode, conv_channels, do_relu):
        super().__init__()

        assert squeeze_mode in ["mean", "4th", "last", "conv"]

        self.configs = {
            "squeeze_mode": squeeze_mode, 
            "conv_channels": conv_channels,
            "do_relu": do_relu
        }
        
        # Downscaling (T) layer 
        self.smush_t = torch.nn.Conv2d(
            in_channels=4, 
            out_channels=1, 
            kernel_size=(T * 3, T * 3), 
            padding="same"
        )

        # Conv layers
        self.convs = nn.Sequential(*create_conv_stack(conv_channels, do_relu=do_relu))

        # Activations
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):  # (B, T=4, H, W)
        fx = self.smush_t(x)  # (B, T=1, H, W)

        fx = self.convs(fx)  # (B, T=1, H, W)

        x = torch.mean(x, dim=1, keepdim=True)
        x = self.sigmoid(x + fx)
        
        return x

# Test code does not run on import
if __name__ == "__main__":
    model = ConvwayNet()
    print(model(torch.zeros((B, T, 69, 420))).shape)