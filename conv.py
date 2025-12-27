import torch, math
from torch import nn
import torch.nn.functional as F


class SymConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.in_c = in_channels
        self.out_c = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.doBias = bias
        self.param_width = math.ceil(kernel_size/2)
        self.params = nn.Parameter(data=torch.randn((self.out_c, self.in_c, (self.param_width**2 + self.param_width)//2)))

    def forward(self, x):
        kernel = torch.zeros((self.out_c, self.in_c, self.kernel_size, self.kernel_size))
        row, col = torch.triu_indices(self.param_width, self.param_width)

        for i in range(4):
            kernel[:, :, row, col] = self.params
            row, col = col, row # diagonal reflection

            kernel[:, :, row, col] = self.params
            row = self.kernel_size-row-1 # paragonal reflection

        return F.conv2d(x, kernel, stride=self.stride, padding=self.padding)


if __name__ == "__main__":
    x = torch.full((1, 2, 9, 9), 2.0)
    conv = SymConv2d(in_channels=2, out_channels=2, kernel_size=5, padding="same")
    x = conv(x)