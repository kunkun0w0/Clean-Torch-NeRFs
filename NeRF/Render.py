import torch
import torch.nn.functional as F
from Sample import uniform_sample_point
from PE import PE


def render_rays(net, rays_o, rays_d, near, far, N_samples, device, noise_std=.0):
    z_vals = uniform_sample_point(near, far, N_samples, device)
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., None]

    # Run network
    pts_flat = torch.reshape(pts, [-1, 3])
    pts_flat = PE(pts_flat, L=10)
    rgb, sigma = net(pts_flat)
    rgb = rgb.view(list(pts.shape[:-1])+[3])
    sigma = sigma.view(list(pts.shape[:-1]))

    # Do volume rendering
    delta = z_vals[..., 1:] - z_vals[..., :-1]
    INF = torch.ones(delta[..., :1].shape, device=device).fill_(1e10)
    delta = torch.cat([delta, INF], -1)
    delta = delta * torch.norm(rays_d, dim=-1, keepdim=True)

    if noise_std > 0.:
        sigma += torch.randn(sigma.size(), device=device) * noise_std

    alpha = 1. - torch.exp(-sigma * delta)
    ones = torch.ones(alpha[..., :1].shape, device=device)
    weights = alpha * torch.cumprod(torch.cat([ones, 1. - alpha], dim=-1), dim=-1)[..., :-1]

    rgb_map = torch.sum(weights[..., None] * rgb, dim=-2)
    depth_map = torch.sum(weights * z_vals, -1)
    acc_map = torch.sum(weights, -1)

    return rgb_map, depth_map, acc_map

