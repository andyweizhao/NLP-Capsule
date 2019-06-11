import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

def squash_v1(x, axis):
    s_squared_norm = (x ** 2).sum(axis, keepdim=True)
    scale = torch.sqrt(s_squared_norm)/ (0.5 + s_squared_norm)
    return scale * x

def dynamic_routing(batch_size, b_ij, u_hat, input_capsule_num):
    num_iterations = 3

    for i in range(num_iterations):
        if True:
            leak = torch.zeros_like(b_ij).sum(dim=2, keepdim=True)
            leaky_logits = torch.cat((leak, b_ij),2)
            leaky_routing = F.softmax(leaky_logits, dim=2)
            c_ij = leaky_routing[:,:,1:,:].unsqueeze(4)
        else:
            c_ij = F.softmax(b_ij, dim=2).unsqueeze(4)
        v_j = squash_v1((c_ij * u_hat).sum(dim=1, keepdim=True), axis=3)
        if i < num_iterations - 1:
            b_ij = b_ij + (torch.cat([v_j] * input_capsule_num, dim=1) * u_hat).sum(3)

    poses = v_j.squeeze(1)
    activations = torch.sqrt((poses ** 2).sum(2))
    return poses, activations


def Adaptive_KDE_routing(batch_size, b_ij, u_hat):
    last_loss = 0.0
    while True:
        if False:
            leak = torch.zeros_like(b_ij).sum(dim=2, keepdim=True)
            leaky_logits = torch.cat((leak, b_ij),2)
            leaky_routing = F.softmax(leaky_logits, dim=2)
            c_ij = leaky_routing[:,:,1:,:].unsqueeze(4)
        else:
            c_ij = F.softmax(b_ij, dim=2).unsqueeze(4)
        c_ij = c_ij/c_ij.sum(dim=1, keepdim=True)
        v_j = squash_v1((c_ij * u_hat).sum(dim=1, keepdim=True), axis=3)
        dd = 1 - ((squash_v1(u_hat, axis=3)-v_j)** 2).sum(3)
        b_ij = b_ij + dd

        c_ij = c_ij.view(batch_size, c_ij.size(1), c_ij.size(2))
        dd = dd.view(batch_size, dd.size(1), dd.size(2))

        kde_loss = torch.mul(c_ij, dd).sum()/batch_size
        kde_loss = np.log(kde_loss.item())

        if abs(kde_loss - last_loss) < 0.05:
            break
        else:
            last_loss = kde_loss
    poses = v_j.squeeze(1)
    activations = torch.sqrt((poses ** 2).sum(2))
    return poses, activations


def KDE_routing(batch_size, b_ij, u_hat):
    num_iterations = 3
    for i in range(num_iterations):
        if False:
            leak = torch.zeros_like(b_ij).sum(dim=2, keepdim=True)
            leaky_logits = torch.cat((leak, b_ij),2)
            leaky_routing = F.softmax(leaky_logits, dim=2)
            c_ij = leaky_routing[:,:,1:,:].unsqueeze(4)
        else:
            c_ij = F.softmax(b_ij, dim=2).unsqueeze(4)

        c_ij = c_ij/c_ij.sum(dim=1, keepdim=True)
        v_j = squash_v1((c_ij * u_hat).sum(dim=1, keepdim=True), axis=3)

        if i < num_iterations - 1:
            dd = 1 - ((squash_v1(u_hat, axis=3)-v_j)** 2).sum(3)
            b_ij = b_ij + dd
    poses = v_j.squeeze(1)
    activations = torch.sqrt((poses ** 2).sum(2))
    return poses, activations

class FlattenCaps(nn.Module):
    def __init__(self):
        super(FlattenCaps, self).__init__()
    def forward(self, p, a):
        poses = p.view(p.size(0), p.size(2) * p.size(3) * p.size(4), -1)
        activations = a.view(a.size(0), a.size(1) * a.size(2) * a.size(3), -1)
        return poses, activations

class PrimaryCaps(nn.Module):
    def __init__(self, num_capsules, in_channels, out_channels, kernel_size, stride):
        super(PrimaryCaps, self).__init__()

        self.capsules = nn.Conv1d(in_channels, out_channels * num_capsules, kernel_size, stride)

        torch.nn.init.xavier_uniform_(self.capsules.weight)

        self.out_channels = out_channels
        self.num_capsules = num_capsules

    def forward(self, x):
        batch_size = x.size(0)
        u = self.capsules(x).view(batch_size, self.num_capsules, self.out_channels, -1, 1)
        poses = squash_v1(u, axis=1)
        activations = torch.sqrt((poses ** 2).sum(1))
        return poses, activations

class FCCaps(nn.Module):
    def __init__(self, args, output_capsule_num, input_capsule_num, in_channels, out_channels):
        super(FCCaps, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_capsule_num = input_capsule_num
        self.output_capsule_num = output_capsule_num

        self.W1 = nn.Parameter(torch.FloatTensor(1, input_capsule_num, output_capsule_num, out_channels, in_channels))
        torch.nn.init.xavier_uniform_(self.W1)

        self.is_AKDE = args.is_AKDE
        self.sigmoid = nn.Sigmoid()


    def forward(self, x, y, labels):
        batch_size = x.size(0)
        variable_output_capsule_num = len(labels)
        W1 = self.W1[:,:,labels,:,:]

        x = torch.stack([x] * variable_output_capsule_num, dim=2).unsqueeze(4)

        W1 = W1.repeat(batch_size, 1, 1, 1, 1)
        u_hat = torch.matmul(W1, x)

        b_ij = Variable(torch.zeros(batch_size, self.input_capsule_num, variable_output_capsule_num, 1)).cuda()

        if self.is_AKDE == True:
            poses, activations = Adaptive_KDE_routing(batch_size, b_ij, u_hat)
        else:
            #poses, activations = dynamic_routing(batch_size, b_ij, u_hat, self.input_capsule_num)
            poses, activations = KDE_routing(batch_size, b_ij, u_hat)
        return poses, activations

