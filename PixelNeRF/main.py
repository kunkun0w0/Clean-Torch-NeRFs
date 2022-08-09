import torch
from tqdm import tqdm
from Dataset import get_dataset
from torch.utils.data import DataLoader
from Network import PixelNeRF
from Render import render_rays
from test_utils import generate_video_nearby
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(999)
np.random.seed(666)
n_train = 3

#############################
# create rays for batch train
#############################
print("Process rays data for training!")
rays_dataset, ref_dataset = get_dataset("../tiny_nerf_data.npz", n_train, device)

#############################
# training parameters
#############################
Batch_size = 2048
rays_loader = DataLoader(rays_dataset, batch_size=Batch_size, drop_last=True, shuffle=True)
print(f"Batch size of rays: {Batch_size}")

bound = (2., 6.)
N_samples = (64, None)
epoch = 100
img_f_ch = 512
lr = 1e-4

#############################
# training
#############################
net = PixelNeRF(img_f_ch).to(device)
mse = torch.nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=lr)

print("Start Training!")
for e in range(epoch):
    with tqdm(total=len(rays_loader), desc=f"Epoch {e+1}", ncols=100) as p_bar:
        for train_rays in rays_loader:
            assert train_rays.shape == (Batch_size, 9)
            rays_o, rays_d, target_rgb = torch.chunk(train_rays, 3, dim=-1)
            rays_od = (rays_o, rays_d)
            rgb, _, __ = render_rays(net, rays_od, bound=bound, N_samples=N_samples, device=device, ref=ref_dataset)
            loss = mse(rgb, target_rgb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            p_bar.set_postfix({'loss': '{0:1.5f}'.format(loss.item())})
            p_bar.update(1)

print('Finish Training!')

print('Start Generating Video!')
net.eval()
generate_video_nearby(net, ref_dataset, bound, N_samples, device, './video/test.mp4')
print('Finish Generating Video!')



