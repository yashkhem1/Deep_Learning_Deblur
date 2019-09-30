import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def real_mse_loss(D_out):
    loss = nn.MSELoss()
    target_tensor = 1.0
    return loss(D_out, torch.tensor(target_tensor).expand_as(D_out).to(device))
    # return torch.mean((D_out-1)**2)

def fake_mse_loss(D_out):
    loss = nn.MSELoss()
    target_tensor = 0.0
    return loss(D_out, torch.tensor(target_tensor).expand_as(D_out).to(device))
    # return torch.mean(D_out**2)

def cycle_consistency_loss  (real_im, reconstructed_im, lambda_weight):
    reconstructed_loss = torch.mean(torch.abs(real_im - reconstructed_im))
    return lambda_weight*reconstructed_loss

def resize2d(img, size):
    if len(list(img.shape)) ==3:
        img = img.reshape(img.shape[0],1,img.shape[1],img.shape[2])
    return F.interpolate(img, size,mode='bilinear')

def multi_scale_loss (real_im, out_list, nlevels):
    loss_total = 0
    for i in range(nlevels):
        height = out_list[i].shape[2]
        weight = out_list[i].shape[3]
        real_i = resize2d(real_im,(height,weight))
        pred_i = out_list[i]
        loss = torch.mean((real_i - pred_i)**2)
        loss_total+= loss

    return loss_total
