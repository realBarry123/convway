import torch, math
from torch import nn
import torch.nn.functional as F


class SymConv2d(nn.Module):

    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.param_width = math.ceil(kernel_size/2)
        self.params = nn.Parameter(data=torch.full(((self.param_width**2 + self.param_width)//2,), 3.0))

    def forward(self, x):
        kernel = torch.empty((self.kernel_size, self.kernel_size))
        indices = torch.triu_indices(self.param_width, self.param_width)
        kernel[indices[0], indices[1]] = self.params
        kernel[indices[0], -indices[1]] = self.params
        kernel[-indices[0], indices[1]] = self.params
        kernel[-indices[0], -indices[1]] = self.params
        indices[0], indices[1] = indices[1], indices[0]
        kernel[indices[0], indices[1]] = self.params
        kernel[indices[0], -indices[1]] = self.params
        kernel[-indices[0], indices[1]] = self.params
        kernel[-indices[0], -indices[1]] = self.params
        kernel = kernel.unsqueeze(0).unsqueeze(0)
        return F.conv2d(x, kernel, padding="same")


if __name__ == "__main__":
    x = torch.full((1, 1, 9, 9), 2.0)
    conv = SymConv2d(1)
    x = conv(x)
    print(x)