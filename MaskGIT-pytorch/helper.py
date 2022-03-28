import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # think of it as a stack of this basic block
        # Conv2D -> GroupNorm -> Swish
        self.block = nn.Sequential(
            GroupNorm(in_channels),
            Swish(),
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            GroupNorm(out_channels),
            Swish(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        )
        # For residual network, in_channel and out_channel must be the same
        # otherwise, we could not add them up together without projection!
        if in_channels != out_channels:
            # projection to align input and output dimension
            # manually convert input to output dimensions
            # CHECKME: Is this the right way to go?
            self.channel_up = nn.Conv2d(in_channels, out_channels, 1, 1, 0)

    def forward(self, x):
        if self.in_channels != self.out_channels:
            return self.block(x) + self.channel_up(x)
        else:
            return x + self.block(x)


class UpSampleBlock(nn.Module):
    def __init__(self, channels):
        super(UpSampleBlock, self).__init__()
        self.conv = nn.Conv2d(channels, channels, 3, 1, 1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.)
        return self.conv(x)


# Downsampling block that reduce resolution of images by half
# CHECKME: Why not maxpooling?
# Used in here: https://github.com/CompVis/taming-transformers/blob/master/taming/modules/diffusionmodules/model.py
class DownSampleBlock(nn.Module):
    def __init__(self, channels):
        super(DownSampleBlock, self).__init__()
        self.conv = nn.Conv2d(channels, channels, 3, 2, 0)

    def forward(self, x):
        pad = (0, 1, 0, 1)
        x = F.pad(x, pad, mode="constant", value=0)
        return self.conv(x)


# Non-local block that captures long-range dependencies vai non-local operations
# Reference: https://paperswithcode.com/paper/non-local-neural-networks
class NonLocalBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = GroupNorm(in_channels)
        self.q = torch.nn.Conv2d(in_channels, in_channels, 1, 1, 0)
        self.k = torch.nn.Conv2d(in_channels, in_channels, 1, 1, 0)
        self.v = torch.nn.Conv2d(in_channels, in_channels, 1, 1, 0)
        self.proj_out = torch.nn.Conv2d(in_channels, in_channels, 1, 1, 0)

    def forward(self, x):
        h_ = self.norm(x)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, h, w = q.shape

        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)
        k = k.reshape(b, c, h * w)
        v = v.reshape(b, c, h * w)

        attn = torch.bmm(q, k)
        attn = attn * (int(c) ** (-0.5))
        attn = F.softmax(attn, dim=2)

        attn = attn.permute(0, 2, 1)
        A = torch.bmm(v, attn)
        A = A.reshape(b, c, h, w)

        A = self.proj_out(A)

        return x + A


class GroupNorm(nn.Module):
    def __init__(self, in_channels):
        super(GroupNorm, self).__init__()
        # Perform Group Normalization, which is a variety of Batch Normalization
        # affine=True => mean and variance are learnable parameters
        self.gn = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

    def forward(self, x):
        return self.gn(x)


# this is the new activation function invented by Google
# f(x) = x sigmoid(x)
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)
