import torch
import torch.nn as nn
import torch.nn.functional as F
from ..modules.conv import Conv, DWConv, RepConv, GhostConv, autopad


class LSKA(nn.Module):
    def __init__(self, dim, k_size):
        super().__init__()

        self.k_size = k_size

        if k_size == 7:
            self.conv0h = nn.Conv2d(dim, dim, kernel_size=(1, 3), stride=(1, 1), padding=(0, (3 - 1) // 2), groups=dim)
            self.conv0v = nn.Conv2d(dim, dim, kernel_size=(3, 1), stride=(1, 1), padding=((3 - 1) // 2, 0), groups=dim)
            self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, 3), stride=(1, 1), padding=(0, 2), groups=dim,
                                            dilation=2)
            self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(3, 1), stride=(1, 1), padding=(2, 0), groups=dim,
                                            dilation=2)
        elif k_size == 11:
            self.conv0h = nn.Conv2d(dim, dim, kernel_size=(1, 3), stride=(1, 1), padding=(0, (3 - 1) // 2), groups=dim)
            self.conv0v = nn.Conv2d(dim, dim, kernel_size=(3, 1), stride=(1, 1), padding=((3 - 1) // 2, 0), groups=dim)
            self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, 5), stride=(1, 1), padding=(0, 4), groups=dim,
                                            dilation=2)
            self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(5, 1), stride=(1, 1), padding=(4, 0), groups=dim,
                                            dilation=2)
        elif k_size == 23:
            self.conv0h = nn.Conv2d(dim, dim, kernel_size=(1, 5), stride=(1, 1), padding=(0, (5 - 1) // 2), groups=dim)
            self.conv0v = nn.Conv2d(dim, dim, kernel_size=(5, 1), stride=(1, 1), padding=((5 - 1) // 2, 0), groups=dim)
            self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, 7), stride=(1, 1), padding=(0, 9), groups=dim,
                                            dilation=3)
            self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(7, 1), stride=(1, 1), padding=(9, 0), groups=dim,
                                            dilation=3)
        elif k_size == 35:
            self.conv0h = nn.Conv2d(dim, dim, kernel_size=(1, 5), stride=(1, 1), padding=(0, (5 - 1) // 2), groups=dim)
            self.conv0v = nn.Conv2d(dim, dim, kernel_size=(5, 1), stride=(1, 1), padding=((5 - 1) // 2, 0), groups=dim)
            self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, 11), stride=(1, 1), padding=(0, 15), groups=dim,
                                            dilation=3)
            self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(11, 1), stride=(1, 1), padding=(15, 0), groups=dim,
                                            dilation=3)
        elif k_size == 41:
            self.conv0h = nn.Conv2d(dim, dim, kernel_size=(1, 5), stride=(1, 1), padding=(0, (5 - 1) // 2), groups=dim)
            self.conv0v = nn.Conv2d(dim, dim, kernel_size=(5, 1), stride=(1, 1), padding=((5 - 1) // 2, 0), groups=dim)
            self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, 13), stride=(1, 1), padding=(0, 18), groups=dim,
                                            dilation=3)
            self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(13, 1), stride=(1, 1), padding=(18, 0), groups=dim,
                                            dilation=3)
        elif k_size == 53:
            self.conv0h = nn.Conv2d(dim, dim, kernel_size=(1, 5), stride=(1, 1), padding=(0, (5 - 1) // 2), groups=dim)
            self.conv0v = nn.Conv2d(dim, dim, kernel_size=(5, 1), stride=(1, 1), padding=((5 - 1) // 2, 0), groups=dim)
            self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, 17), stride=(1, 1), padding=(0, 24), groups=dim,
                                            dilation=3)
            self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(17, 1), stride=(1, 1), padding=(24, 0), groups=dim,
                                            dilation=3)

        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0h(x)
        attn = self.conv0v(attn)
        attn = self.conv_spatial_h(attn)
        attn = self.conv_spatial_v(attn)
        attn = self.conv1(attn)
        return u * attn


class LSKA_Attention(nn.Module):
    def __init__(self, d_model, k_size):
        super().__init__()

        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = LSKA(d_model, k_size)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x


class LSKA_AttentionC3(nn.Module):
    """Rep C3."""

    def __init__(self, c1, c2, n=3, e=1.0):
        """Initialize CSP Bottleneck with a single convolution using input channels, output channels, and number."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.m = nn.Sequential(*[LSKA_Attention(c_, 11) for _ in range(n)])
        self.cv3 = Conv(c_, c2, 1, 1) if c_ != c2 else nn.Identity()

    def forward(self, x):
        """Forward pass of RT-DETR neck layer."""
        return self.cv3(self.m(self.cv1(x)) + self.cv2(x))