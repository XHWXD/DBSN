import torch
import torch.nn as nn
import torch.nn.functional as F

# L2Loss is mainly used to pre-train mu
class L2Loss(nn.Module):
    def __init__(self):
        super(L2Loss,self).__init__()
        self.L2=nn.MSELoss(reduction='mean')
    def forward(self, target, mu):
        loss = 0
        target = target.detach()
        loss = self.L2(target,mu)
        return loss

# To pre-train sigma_mu & sigma_n
class MAPLoss_Pretrain(nn.Module):
    def __init__(self):
        super(MAPLoss_Pretrain,self).__init__()
    def forward(self, target, mu, sigma_mu, sigma_n, sigma_y):
        loss = 0
        target = target.detach()
        mu = mu.detach()
        t1 = ((target - mu) ** 2) / sigma_y 
        t2 = sigma_y.log()
        loss = t1 + t2
        loss = loss.mean()
        return loss

# To finetune the framework
class MAPLoss(nn.Module):
    def __init__(self):
        super(MAPLoss,self).__init__()
    def forward(self, target, mu, sigma_mu, sigma_n, sigma_y):
        loss = 0
        target = target.detach()
        t1 = ((target - mu) ** 2) / sigma_y 
        t2 = sigma_y.log()
        # t3 = 0.1*sigma_n.sqrt() 
        loss = t1 + t2 # - t3 # t3 for AWGN only
        loss = loss.mean()
        if t1.max() > 1e+8:
            loss.data.zero_()
        return loss

# To pre-train sigma_mu & sigma_n
class DBSNLoss_Pretrain(nn.Module):
    def __init__(self):
        super(DBSNLoss_Pretrain,self).__init__()
    def forward(self, target, mu, sigma_mu, sigma_n, sigma_y):
        loss = 0
        eps = 1e-6
        target = target.detach()
        mu = mu.detach()
        t1 = ((target - mu) ** 2) / sigma_y
        t2 = (sigma_n.clamp(eps)).log()
        t3 = sigma_mu / (sigma_n).clamp(eps)
        loss = t1 + t2 + t3
        loss = loss.mean()
        if t1.max() > 1e+8 or t3.max()> 1e+8:
            loss.data.zero_()
        return loss

# To finetune the framework
class DBSNLoss(nn.Module):
    def __init__(self):
        super(DBSNLoss,self).__init__()
    def forward(self, target, mu, sigma_mu, sigma_n, sigma_y):
        loss = 0
        eps = 1e-6
        target = target.detach()
        t1 = ((target - mu) ** 2) / sigma_y
        t2 = (sigma_n.clamp(eps)).log()
        t3 = sigma_mu / (sigma_n).clamp(eps)
        loss = t1 + t2 + t3
        loss = loss.mean()
        if t1.max() > 1e+8 or t3.max()> 1e+8:
            loss.data.zero_()
        return loss