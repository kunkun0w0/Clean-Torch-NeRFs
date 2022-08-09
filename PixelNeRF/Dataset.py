import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import numpy as np
from ImageEncoder import ImageEncoder


def sample_rays_np(H, W, f, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i - W * .5 + 0.5) / f, -(j - H * .5 + 0.5) / f, -np.ones_like(i)], -1)
    rays_d = np.sum(dirs[..., None, :] * c2w[:3, :3], -1)
    rays_o = np.broadcast_to(c2w[:3, -1], np.shape(rays_d))
    return rays_o, rays_d


def get_dataset(data_dir, n, device):
    data = np.load(data_dir)
    images = data['images']
    poses = data['poses']
    focal = data['focal']
    train_list = shuffle_id(images.shape[0], n)
    H, W = images.shape[1:3]
    rays = create_ray_batches(images, poses, train_list, H, W, focal, device)

    train_images = torch.tensor(images[train_list], device=device).permute(0, 3, 1, 2)
    encoder = ImageEncoder().to(device).eval()
    with torch.no_grad():
        reference_feature = encoder(train_images)
        # reference_feature => tensor(n, 512, 50, 50)

    return RaysDataset(rays), ReferenceDataset(reference_feature, poses[train_list], focal, H)


def shuffle_id(n, k):
    train_list = np.arange(n)
    np.random.shuffle(train_list)
    train_list = train_list[:k]
    return train_list


def create_ray_batches(images, poses, train_list, H, W, f, device):
    print("Create Ray batches!")
    rays_o_list = list()
    rays_d_list = list()
    rays_rgb_list = list()
    for i in train_list:
        img = images[i]
        pose = poses[i]
        rays_o, rays_d = sample_rays_np(H, W, f, pose)
        rays_o_list.append(rays_o.reshape(-1, 3))
        rays_d_list.append(rays_d.reshape(-1, 3))
        rays_rgb_list.append(img.reshape(-1, 3))
    rays_o_npy = np.concatenate(rays_o_list, axis=0)
    rays_d_npy = np.concatenate(rays_d_list, axis=0)
    rays_rgb_npy = np.concatenate(rays_rgb_list, axis=0)
    rays = torch.tensor(np.concatenate([rays_o_npy, rays_d_npy, rays_rgb_npy], axis=1), device=device)
    return rays


class RaysDataset(Dataset):
    def __init__(self, rays):
        self.rays = rays

    def __len__(self):
        return self.rays.shape[0]

    def __getitem__(self, idx):
        return self.rays[idx]


class ReferenceDataset:
    def __init__(self, reference, c2w, f, img_size):
        self.reference = reference
        self.scale = (img_size / 2) / f
        self.n = c2w.shape[0]
        self.R_t = torch.tensor(c2w[:, :3, :3], device=reference.device).permute(0, 2, 1)
        self.camera_pos = torch.tensor(c2w[:, :3, -1], device=reference.device)
        self.c2w = c2w
        self.img_size = img_size
        self.f = f

    @torch.no_grad()
    def feature_matching(self, pos):
        n_rays, n_samples, _ = pos.shape
        pos = pos.unsqueeze(dim=0).expand([self.n, n_rays, n_samples, 3])
        camera_pos = self.camera_pos[:, None, None, :]
        camera_pos = camera_pos.expand_as(pos)
        ref_pos = torch.einsum("kij,kbsj->kbsi", self.R_t, pos-camera_pos)
        uv_pos = ref_pos[..., :-1] / ref_pos[..., -1:] / self.scale
        uv_pos[..., 1] *= -1.0
        return F.grid_sample(self.reference, uv_pos, align_corners=True, padding_mode="border")

