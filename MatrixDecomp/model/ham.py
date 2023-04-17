import torch
from torch import nn
from torch.nn import functional as F

# NMF matrix decomposition
class NMF(nn.Module):
    def __init__(self, S = 1, D = 512, R = 64, steps = 6):
        super(NMF, self).__init__()
        self.S = S
        self.D = D
        self.R = R
        self.steps = steps
        D_matrix = torch.rand(1 * self.S, D, self.R)
        D_matrix = F.normalize(D_matrix, dim = 1)
        self.register_buffer('D_matrix', D_matrix)

    @torch.no_grad()
    def single_step(self, x, C_matrix, D_matrix):
        
        top = torch.einsum('bdn, bdr -> bnr', x, D_matrix)

        bottom = torch.einsum('bnr, bdr -> bdn', C_matrix, D_matrix)
        bottom = torch.einsum('bdn, bdr -> bnr', bottom, D_matrix)

        C_matrix = C_matrix * (top / bottom + 1e-6)

        top = torch.einsum('bdn, bnr -> bdr', x, C_matrix)

        bottom = torch.einsum('bnr, bdr -> bdn', C_matrix, D_matrix)
        bottom = torch.einsum('bdn, bnr -> bdr', bottom, C_matrix)

        D_matrix = D_matrix * (top / bottom + 1e-6)

        return D_matrix, C_matrix
    

    @torch.no_grad()
    def update(self, D_matrix):
        update = D_matrix.mean(dim = 0)
        self.D_matrix += 0.9 * (update - self.D_matrix)
        self.D_matrix = F.normalize(self.D_matrix, dim = 1)

    
    
    def forward(self, x):
        B, C, H, W = x.shape

        D = C // self.S
        N = H * W
        x = x.view(B * self.S, D, N)

        D_matrix = self.D_matrix.repeat(B, 1, 1)

        with torch.no_grad():
            C_matrix = torch.einsum('bdn, bdr -> bnr', x, D_matrix)

            C_matrix = F.softmax(C_matrix, dim = -1)
        
            for i in range(self.steps):
                D_matrix, C_matrix = self.single_step(x, C_matrix, D_matrix)
        
        
        x = torch.einsum('bnr, bdr -> bdn', C_matrix, D_matrix)

        x = x.view(B, C, H, W)

        D_matrix = D_matrix.view(B, self.S, D, self.R)

        if not self.training:
            self.update(D_matrix)

        return x
    


class VQ(nn.Module):
    def __init__(self, S = 1, D = 512, R = 64, steps = 6):
        super(NMF, self).__init__()
        self.S = S
        self.D = D
        self.R = R
        self.steps = steps
        D_matrix = torch.rand(1 * self.S, D, self.R)
        D_matrix = F.normalize(D_matrix, dim = 1)
        self.register_buffer('D_matrix', D_matrix)
    
    @torch.no_grad()
    def update(self, D_matrix):
        update = D_matrix.mean(dim = 0)
        self.D_matrix += 0.9 * (update - self.D_matrix)
        self.D_matrix = F.normalize(self.D_matrix, dim = 1)
    
    @torch.no_grad()
    def single_step(self, x, D_matrix):
        std_x = F.normalize(x, dim = 1)

        std_D_matrix = F.normalize(D_matrix, dim = 1, eps = 1e-6)

        C_matrix = torch.einsum('bdn, bdr -> bnr', std_x, std_D_matrix)

        C_matrix = F.softmax(C_matrix, dim = -1)

        C_matrix = C_matrix / (C_matrix.sum(dim = -1, keepdim = True) + 1e-6)

        D_matrix = torch.einsum('bnr, bdr -> bdn', C_matrix, std_x)

        return D_matrix, C_matrix

    def forward(self, x):
        B, C, H, W = x.shape

        D = C // self.S
        N = H * W
        x = x.view(B * self.S, D, N)

        D_matrix = self.D_matrix.repeat(B, 1, 1)

        with torch.no_grad():
            C_matrix = torch.einsum('bdn, bdr -> bnr', x, D_matrix)

            C_matrix = F.softmax(C_matrix, dim = -1)
        
            for i in range(self.steps):
                D_matrix, C_matrix = self.single_step(x, C_matrix, D_matrix)
        
        
        x = torch.einsum('bnr, bdr -> bdn', C_matrix, D_matrix)

        x = x.view(B, C, H, W)

        D_matrix = D_matrix.view(B, self.S, D, self.R)

        if not self.training:
            self.update(D_matrix)

        return x