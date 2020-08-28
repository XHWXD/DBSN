import torch
import torch.nn as nn
import torch.nn.functional as F
from util.eig_decompose_3x3 import eigs_comp, eigs_vec_comp

eps = 1e-6

# Tp pre-train mu
class L2Loss(nn.Module):
    def __init__(self):
        super(L2Loss,self).__init__()
        self.L2=nn.MSELoss(reduction='sum')
    def forward(self, target, output):
        loss = 0
        target = target.detach()
        loss = self.L2(target,output)
        return loss


# To pre-train sigma_mu and sigma_n for G/PG
class MAPLoss_Pretrain(nn.Module):
    def __init__(self):
        super(MAPLoss_Pretrain,self).__init__()
    def forward(self, target, mu, sigma_mu, sigma_n, sigma_y):
        loss = 0
        target = target.detach()
        mu = mu.detach()
        batch, c, m, n=target.shape
        T = (target - mu).permute(0,2,3,1).unsqueeze(-1)
        t1 = 0.5 * ( T.transpose(3,4) @ torch.inverse(sigma_y) @ T ).squeeze(-1).squeeze(-1) # NHW
        t2 = 0.5 * torch.log(torch.det(sigma_y)) # NHW
        loss = t1 + t2 # NHW
        loss = loss.mean()
        if t1.max() > 1e+8:
            loss.data.zero_()
        return loss

# To finetune mu, sigma_mu and sigma_n for G/PG/MG
class MAPLoss(nn.Module):
    def __init__(self):
        super(MAPLoss,self).__init__()
    def forward(self, target, mu, sigma_mu, sigma_n, sigma_y):
        loss = 0
        target = target.detach()
        batch, c, m, n=target.shape
        T = (target - mu).permute(0,2,3,1).unsqueeze(-1)
        t1 = 0.5 * ( T.transpose(3,4) @ torch.inverse(sigma_y) @ T ).squeeze(-1).squeeze(-1) # NHW
        dets = torch.det(sigma_y)
        dets = 0.5 * torch.log(dets.clamp(eps)) # NHW
        loss = t1 + dets # NHW
        loss = loss.mean()
        if t1.max() > 1e+7:
            loss.data.zero_()
        return loss

# To pre-train sigma_n and sigma_mu for G/MG
class DBSNLoss_Pretrain(nn.Module):
    def __init__(self):
        super(DBSNLoss_Pretrain,self).__init__()
    def forward(self, target, mu, sigma_mu, sigma_n, sigma_y):
        loss = 0
        target = target.detach()
        mu = mu.detach()
        batch, c, m, n=target.shape
        I_matrix = eps*torch.eye(3,device='cuda').repeat(batch,m,n,1,1)
        T = (target - mu).permute(0,2,3,1).unsqueeze(-1)
        t1 = 0.5 * ( T.transpose(3,4) @ torch.inverse(sigma_y) @ T ).squeeze(-1).squeeze(-1) # NHW
        t2 = 0.5 * torch.log(torch.det(sigma_n).clamp(eps)) # NHW
        tmp = torch.inverse(sigma_n+I_matrix) @ sigma_mu
        eig_values = eigs_comp(tmp.view(batch*m*n,3,3)) #?33
        eig_values = eig_values.sum(dim=(1,2)).view(batch,m,n) #NHW
        t3 = 0.5*eig_values
        loss = t1 + t2 + t3 # NHW
        loss = loss.mean()
        if t1.max() > 1e+7:
            loss.data.zero_()
        return loss

# To finetune mu, sigma_mu and sigma_n for G
class DBSNLoss(nn.Module):
    def __init__(self):
        super(DBSNLoss,self).__init__()
    def forward(self, target, mu, sigma_mu, sigma_n, sigma_y):
        loss = 0
        target = target.detach()
        batch, c, m, n=target.shape
        I_matrix = eps*torch.eye(3,device='cuda').repeat(batch,m,n,1,1)
        T = (target - mu).permute(0,2,3,1).unsqueeze(-1)
        t1 = 0.5 * ( T.transpose(3,4) @ torch.inverse(sigma_y) @ T ).squeeze(-1).squeeze(-1) # NHW
        t2 = 0.5 * torch.log(torch.det(sigma_n).clamp(eps)) # NHW
        tmp = torch.inverse(sigma_n+I_matrix) @ sigma_mu
        eig_values = eigs_comp(tmp.view(batch*m*n,3,3)) #?33
        eig_values = eig_values.sum(dim=(1,2)).view(batch,m,n) #NHW
        t3 = 0.5*eig_values
        loss = t1 + t2 + t3 # NHW
        loss = loss.mean()
        if t1.max() > 1e+7:
            loss.data.zero_()
        return loss