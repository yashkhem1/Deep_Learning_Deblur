import torch
import sys
import math
import random
import cv2


def dice_coeff(inputs, target):
    eps = 1e-7
    coeff = 0
    for i in range(inputs.shape[0]):
        iflat = inputs[i, :, :, :].view(-1)
        tflat = target[i, :, :, :].view(-1)
        intersection = torch.dot(iflat, tflat)
        # print(intersection)
        # print(iflat.sum())
        # print(tflat.sum())
        coeff += (2. * intersection) / (iflat.sum() + tflat.sum() + eps)
    return coeff / (inputs.shape[0])


def normalization(X):
    return X / 127.5 - 1.0

def scale(X):
    return 2*X-1

def denormalize(X, flag=None):
    if flag is None:
        return (X + 1.0) * 127.5
    else:
        return (X + 1.0) / 2.0

def dice_loss(inputs, target):
    return 1 - dice_coeff(inputs, target)



def ab_rel_diff(ip, tar):
    diff = 0
    for i in range(ip.shape[0]):
        iflat = ip[i, :, :, :].view(-1)
        tflat = tar[i, :, :, :].view(-1)
        absm = abs(iflat-tflat)/tflat
        diff+=absm.sum()/len(iflat)
    return diff/ip.shape[0]

def sq_rel_diff(ip, tar):
    diff = 0
    for i in range(ip.shape[0]):
        iflat = ip[i, :, :, :].view(-1)
        tflat = tar[i, :, :, :].view(-1)
        absm = abs(iflat-tflat)**2/tflat
        diff += absm.sum()/len(iflat)
    return diff/ip.shape[0]

def rms_linear(ip, tar):
    diff = 0
    for i in range(ip.shape[0]):
        iflat = ip[i, :, :, :].view(-1)
        tflat = tar[i, :, :, :].view(-1)
        absm = abs(iflat-tflat)**2
        absm = 255*absm
        diff += (absm.sum()/len(iflat))**0.5
    return diff/ip.shape[0]


class ImagePool():

    def __init__(self, pool_size):

        self.pool_size = pool_size
        if self.pool_size > 0:  # create an empty pool
            self.num_imgs = 0
            self.images = []

    def query(self, images):

        if self.pool_size == 0:  # if the buffer size is 0, do nothing
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.pool_size:   # if the buffer is not full; keep inserting current images to the buffer
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:  # by 50% chance, the buffer will return a previously stored image, and insert the current image into the buffer
                    random_id = random.randint(0, self.pool_size - 1)  # randint is inclusive
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:       # by another 50% chance, the buffer will return the current image
                    return_images.append(image)
        return_images = torch.cat(return_images, 0)   # collect all the images and return
        return return_images
