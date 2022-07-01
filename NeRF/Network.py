import torch.nn as nn
import torch
import torch.nn.functional as F


class NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=60, input_ch_views=24, skip=4, use_view_dirs=True):
        """
        :param D: depth of MLP backbone
        :param W: width of MLP backbone
        :param input_ch: encoded RGB input's channels
        :param input_ch_views: encoded view input's channels
        :param skip: when skip connect
        :param use_view_dirs: use view-dependent
        """
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skip = skip
        self.use_view_dirs = use_view_dirs

        self.net = nn.ModuleList([nn.Linear(input_ch, W)])
        for i in range(D-1):
            if i == skip:
                self.net.append(nn.Linear(W + input_ch, W))
            else:
                self.net.append(nn.Linear(W, W))

        self.alpha_linear = nn.Linear(W, 1)
        self.feature_linear = nn.Linear(W, W)
        if use_view_dirs:
            self.proj = nn.Linear(W + input_ch_views, W // 2)
        else:
            self.proj = nn.Linear(W, W // 2)
        self.rgb_linear = nn.Linear(W // 2, 3)

    def forward(self, input_pts, input_views=None):
        h = input_pts.clone()
        for i, _ in enumerate(self.net):
            h = F.relu(self.net[i](h))
            if i == self.skip:
                h = torch.cat([input_pts, h], -1)

        alpha = F.relu(self.alpha_linear(h))
        feature = self.feature_linear(h)

        if self.use_view_dirs:
            h = torch.cat([feature, input_views], -1)

        h = F.relu(self.proj(h))
        rgb = torch.sigmoid(self.rgb_linear(h))

        return rgb, alpha
