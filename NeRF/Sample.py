import torch
import torch.nn.functional as F
import numpy as np


def sample_rays_np(H, W, f, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i - W * .5) / f, -(j - H * .5) / f, -np.ones_like(i)], -1)
    rays_d = np.sum(dirs[..., None, :] * c2w[:3, :3], -1)
    rays_o = np.broadcast_to(c2w[:3, -1], np.shape(rays_d))
    return rays_o, rays_d


# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, N_samples):
    # 归一化 w 求 pdf
    pdf = F.normalize(weights, p=1, dim=-1)
    # 前缀和求 cdf
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)

    # 进行 uniform sampling
    # 最后 u => tensor(batch, bins)
    u = torch.rand(list(cdf.shape[:-1]) + [N_samples]).contiguous()

    # 逆采样
    ids = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(ids - 1), ids - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(ids), ids)
    ids_g = torch.stack([below, above], -1)
    # ids_g => (batch, N_samples, 2)
    # 代表每个采样点所在的区间左右端点序号
    # ids => tensor([[3, 2, 2, 2]])
    # ids_g => tensor([[[2, 3],
    #                   [1, 2],
    #                   [1, 2],
    #                   [1, 2]]])

    matched_shape = [ids_g.shape[0], ids_g.shape[1], cdf.shape[-1]]
    # matched_shape : [batch, N_samples, bins]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, ids_g)
    # 取 cdf 中对应区间的左右端点 cdf 值
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, ids_g)
    # 取 bins 中定义域对应的左右端点值, 其中 bins 是世界坐标系中对应的 ray 分段区间

    # 求采样点的世界坐标
    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


def uniform_sample_point(tn, tf, N_samples, device):
    k = torch.rand([N_samples], device=device) / float(N_samples)
    pt_value = torch.linspace(0.0, 1.0, N_samples + 1, device=device)[:-1]
    pt_value += k
    return tn + (tf - tn) * pt_value


if __name__ == "__main__":
    a = torch.eye(4)
    b = torch.randn([4, 1])
    c1 = torch.matmul(a, b)
    c2 = b[:, None, :]
    print(c1, c2.shape)
    pass
