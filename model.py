import torch

B = 32
T = 4

class ConvwayNet(torch.nn.Module):

    def __init__(self):
        super().__init__()
        
        # Downscaling (T) layer 
        self.conv1 = torch.nn.Conv2d(
            in_channels=4, 
            out_channels=1, 
            kernel_size=(4 * 3, 4 * 3), 
            padding="same"
        )

        # Conv layer
        self.conv2 = torch.nn.Conv2d(
            in_channels=1, 
            out_channels=1, 
            kernel_size=(4 * 3, 4 * 3), 
            padding="same"
        )

    def forward(self, x):
        # (B, T=4, H, W)
        x = self.conv1(x)
        # (B, T=1, H, W)
        x = self.conv2(x)
        # (B, T=1, H, W)
        return x

# Test code does not run on import
if __name__ == "__main__":
    model = ConvwayNet()
    print(model(torch.zeros((B, T, 69, 420))).shape)