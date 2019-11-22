import torch
import sys
import math
import random
import cv2


def normalization(X):
    return X / 127.5 - 1.0

def scale_down(X):
    return X/255.0

def scale_up(X):
    out = torch.clamp(X,0.0,1.0)
    return out*255.0


def scale(X):
    return 2*X-1

def denormalize(X, flag=None):
    if flag is None:
        return (X + 1.0) * 127.5
    else:
        return (X + 1.0) / 2.0

