import torch


# Positional encoding (section 5.1)
def PE(x, L):
    pe = list()
    for i in range(L):
        for fn in [torch.sin, torch.cos]:
            pe.append(fn(2. ** i * x))
    return torch.cat(pe, -1)
