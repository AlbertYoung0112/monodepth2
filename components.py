from torch import nn
import math


class Identity(nn.Module):
    # a dummy identity module
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def img2pc_bridge(in_channel, out_channel, kernel_size):
    return nn.Sequential(
        nn.Conv2d(in_channel, in_channel, kernel_size=kernel_size,
                  padding=(0, kernel_size // 2), groups=in_channel),
        nn.Flatten(start_dim=2),
        nn.BatchNorm1d(in_channel),
        nn.ReLU6(inplace=True),
        nn.Conv1d(in_channel, out_channel, kernel_size=1),
        nn.BatchNorm1d(out_channel),
        nn.ReLU6(inplace=True),
    )


def depth_conv_1d(in_channel, out_channel, kernel_size, stride):
    return nn.Sequential(
        nn.Conv1d(in_channel, in_channel, kernel_size=kernel_size, stride=stride,
                  groups=in_channel, bias=False, padding=1),
        nn.BatchNorm1d(in_channel),
        nn.ReLU6(inplace=True),
        nn.Conv1d(in_channel, out_channel, kernel_size=1, stride=1, bias=False),
        nn.BatchNorm1d(out_channel),
        nn.ReLU6(inplace=True)
    )


def depthwise(in_channels, kernel_size):
    padding = (kernel_size-1) // 2
    assert 2*padding == kernel_size-1, "parameters incorrect. kernel={}, padding={}".format(kernel_size, padding)
    return nn.Sequential(
        nn.Conv2d(in_channels,in_channels,kernel_size,stride=1,padding=padding,bias=False,groups=in_channels),
        nn.BatchNorm2d(in_channels),
        nn.ReLU(inplace=True),
    )


def pointwise(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels,out_channels,1,1,0,bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


def weights_init(m):
    # Initialize kernel weights with Gaussian distributions
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
