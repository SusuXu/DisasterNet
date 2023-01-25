import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.optim as optim



class Planar(nn.Module):
    def __init__(self):
        super(Planar, self).__init__()
        self.h = nn.Tanh()

    def forward(self, z, u, w, b):
        z = z.unsqueeze(2)
        prod = torch.bmm(w, z) + b
        f_z = z + u * self.h(prod)
        f_z = f_z.squeeze(2)
        psi = w * (1 - self.h(prod) ** 2)  
        log_det_jacobian = torch.log(torch.abs(1 + torch.bmm(psi, u)))
        log_det_jacobian = log_det_jacobian.squeeze(2).squeeze(1)

        return f_z, log_det_jacobian
