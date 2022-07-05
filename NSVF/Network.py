import torch.nn as nn
import torch
import torch.nn.functional as F


class NSVF(nn.Module):
    def __init__(self, W=256, input_ch=416, input_ch_views=24, init_voxels=10):
        super(NSVF, self).__init__()
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views

        self.net = list()
        self.net.append(nn.Linear(input_ch, W))
        self.net.append(nn.ReLU(inplace=True))
        self.net.append(nn.Linear(W, W))
        self.net.append(nn.ReLU(inplace=True))
        self.net.append(nn.Linear(W, W))
        self.net = nn.Sequential(*self.net)

        self.alpha_linear = list()
        self.alpha_linear.append(nn.Linear(W, W // 2))
        self.alpha_linear.append(nn.ReLU(inplace=True))
        self.alpha_linear.append(nn.Linear(W // 2, 1))
        self.alpha_linear = nn.Sequential(*self.alpha_linear)

        self.rgb_linear = list()
        self.rgb_linear.append(nn.Linear(W + input_ch_views, W))
        self.rgb_linear.append(nn.ReLU(inplace=True))
        for i in range(3):
            self.rgb_linear.append(nn.Linear(W, W))
            self.rgb_linear.append(nn.ReLU(inplace=True))
        self.rgb_linear.append(nn.Linear(W, 1))
        self.rgb_linear.append(nn.Sigmoid())
        self.rgb_linear = nn.Sequential(*self.rgb_linear)

        self.voxels = nn.Parameter(torch.rand([init_voxels, init_voxels, init_voxels], requires_grad=True))

    def forward(self, input_pts, input_views):
        f = self.net(input_pts)
        alpha = self.alpha_linear(f)
        rgb = self.rgb_linear(torch.cat([f, input_views], dim=-1))
        return rgb, alpha

    @torch.no_grad()
    def prune_voxels(self):
        pass

    @torch.no_grad()
    def split_voxels(self):
        pass

