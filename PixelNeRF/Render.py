import torch


def render_rays(net, rays, bound, N_samples, device, ref, noise_std=.0):
    rays_o, rays_d = rays
    near, far = bound
    N_c, N_f = N_samples

    # coarse sampling
    z_vals = get_coarse_query_points(near, far, N_c, device)
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., None]

    # get projection feature
    f_projection = ref.feature_matching(pts)

    # neural rendering
    rgb, w = get_rgb_w(net, pts, rays_d, z_vals, f_projection, device, noise_std)

    rgb_map = torch.sum(w[..., None] * rgb, dim=-2)
    depth_map = torch.sum(w * z_vals, -1)
    acc_map = torch.sum(w, -1)

    return rgb_map, depth_map, acc_map


def get_coarse_query_points(tn, tf, N_samples, device):
    k = torch.rand([N_samples], device=device) / float(N_samples)
    pt_value = torch.linspace(0.0, 1.0, N_samples + 1, device=device)[:-1]
    pt_value += k
    return tn + (tf - tn) * pt_value


def get_rgb_w(net, pts, rays_d, z_vals, ref_feature, device, noise_std=.0):
    # pts => tensor(N_Rays, N_Samples, 3)
    # rays_d => tensor(N_Rays, 3)
    # ref_feature => tensor(N_References, C, N_Rays, N_Samples)
    # Run network
    rgb, sigma = net(ref_feature, pts, rays_d)
    rgb = rgb.view(list(pts.shape[:-1]) + [3])
    sigma = sigma.view(list(pts.shape[:-1]))

    # get the interval
    delta = z_vals[..., 1:] - z_vals[..., :-1]
    INF = torch.ones(delta[..., :1].shape, device=device).fill_(1e10)
    delta = torch.cat([delta, INF], -1)
    delta = delta * torch.norm(rays_d, dim=-1, keepdim=True)

    # add noise to sigma
    if noise_std > 0.:
        sigma += torch.randn(sigma.size(), device=device) * noise_std

    # get weights
    alpha = 1. - torch.exp(-sigma * delta)
    ones = torch.ones(alpha[..., :1].shape, device=device)
    weights = alpha * torch.cumprod(torch.cat([ones, 1. - alpha], dim=-1), dim=-1)[..., :-1]

    return rgb, weights
