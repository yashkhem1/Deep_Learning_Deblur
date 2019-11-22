import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def resize2d(img, size):
    if len(list(img.shape)) ==3:
        img = img.reshape(img.shape[0],1,img.shape[1],img.shape[2])
    return F.interpolate(img, size,mode='bilinear')

def multi_scale_loss (real_im, out_list, nlevels):
    # loss_total = 0
    for i in range(nlevels):
        height = out_list[i].shape[2]
        weight = out_list[i].shape[3]
        real_i = resize2d(real_im,(height,weight))
        pred_i = out_list[i]
        if i==0:
            loss = F.mse_loss(real_i,pred_i)
        else:
            loss += F.mse_loss(real_i,pred_i)


    return loss
