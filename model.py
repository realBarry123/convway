import torch

B = 32
T = 4

class ConvwayNet(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = torch.nn.Conv2d(
            in_channels=4, 
            out_channels=4, 
            kernel_size=(T * 3, T * 3), 
            padding="same"
        )
        
        # Downscaling (T) layer 
        self.smush_t = torch.nn.Conv2d(
            in_channels=4, 
            out_channels=1, 
            kernel_size=(T * 3, T * 3), 
            padding="same"
        )

        # Conv layer
        self.conv2 = torch.nn.Conv2d(
            in_channels=1, 
            out_channels=1, 
            kernel_size=(T * 3, T * 3), 
            padding="same"
        )

        # Activations
        self.relu1 = torch.nn.LeakyReLU()
        self.relu2 = torch.nn.LeakyReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):  # (B, T=4, H, W)
        x = self.conv1(x)
        x = self.relu1(x)

        x = self.smush_t(x)  # (B, T=1, H, W)
        fx = self.relu2(x)

        fx = self.conv2(fx)  # (B, T=1, H, W)
        # fx = self.leaky_relu(fx)
        x = self.sigmoid(x + fx)
        
        return x

# Test code does not run on import
if __name__ == "__main__":
    model = ConvwayNet()
    print(model(torch.zeros((B, T, 69, 420))).shape)