import torch
import matplotlib.pyplot as plt
import numpy as np
from Network import NeRF
from Sample import sample_rays_np
from Render import render_rays
from tqdm import tqdm


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
torch.manual_seed(999)
np.random.seed(666)
n_train = 100

#############################
# load data
#############################
data = np.load('tiny_nerf_data.npz')
images = data['images']
poses = data['poses']
focal = data['focal']
H, W = images.shape[1:3]
print("images.shape:", images.shape)
print("poses.shape:", poses.shape)
print("focal:", focal)

test_img, test_pose = images[101], poses[101]
images = images[:n_train]
poses = poses[:n_train]

# plt.imshow(test_img)
# plt.show()

#############################
# create rays for batch train
#############################
print("Process rays data!")
rays_o_list = list()
rays_d_list = list()
rays_rgb_list = list()

for i in range(n_train):
    img = images[i]
    pose = poses[i]
    rays_o, rays_d = sample_rays_np(H, W, focal, pose)

    rays_o_list.append(rays_o.reshape(-1, 3))
    rays_d_list.append(rays_d.reshape(-1, 3))
    rays_rgb_list.append(img.reshape(-1, 3))

rays_o_npy = np.concatenate(rays_o_list, axis=0)
rays_d_npy = np.concatenate(rays_d_list, axis=0)
rays_rgb_npy = np.concatenate(rays_rgb_list, axis=0)
rays = torch.tensor(np.concatenate([rays_o_npy, rays_d_npy, rays_rgb_npy], axis=1), device=device)

#############################
# train
#############################
N = rays.shape[0]
Batch_size = 4096
iterations = N // Batch_size
print(f"There are {iterations} rays batches and each batch contains {Batch_size} rays")

N_samples = 64
epoch = 2
psnrs = []
e_nums = []

net = NeRF(use_view_dirs=False).to(device)
optimizer = torch.optim.Adam(net.parameters(), 5e-4)
mse = torch.nn.MSELoss()

for e in range(epoch):
    # create iteration for training
    rays = rays[torch.randperm(N), :]
    train_iter = iter(torch.split(rays, Batch_size, dim=0))

    # render + mse
    with tqdm(total=iterations, desc=f"Epoch {e+1}", ncols=100) as p_bar:
        for i in range(iterations):
            train_rays = next(train_iter)
            assert train_rays.shape == (Batch_size, 9)

            rays_o, rays_d, target_rgb = torch.chunk(train_rays, 3, dim=-1)
            rgb, _, __ = render_rays(net, rays_o, rays_d, near=2., far=6., N_samples=N_samples, device=device)

            loss = mse(rgb, target_rgb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            p_bar.set_postfix({'loss': '{0:1.5f}'.format(loss.item())})
            p_bar.update(1)

    rays_o, rays_d = sample_rays_np(H, W, focal, test_pose)
    rays_o = torch.tensor(rays_o, device=device)
    rays_d = torch.tensor(rays_d, device=device)

    rgb, depth, acc = render_rays(net, rays_o, rays_d, near=2., far=6., N_samples=N_samples, device=device)
    loss = mse(rgb, torch.tensor(test_img, device=device))
    psnr = -10. * torch.log(loss).item() / torch.log(torch.tensor([10.]))
    print(f"PSNR={psnr.item()}")
    psnrs.append(psnr.numpy())
    e_nums.append(e+1)

    plt.figure(figsize=(10, 4))
    plt.subplot(121)
    plt.imshow(rgb.cpu().detach().numpy())
    plt.title(f'Iteration: {i + 1}')
    plt.subplot(122)
    plt.plot(e_nums, psnrs)
    plt.title('PSNR')
    plt.show()

print('Done')
