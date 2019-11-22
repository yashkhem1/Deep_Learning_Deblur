import torch
import numpy as np
import torch.utils.data
import glob
import cv2
import random
import torchvision.transforms.functional as TF

class GOPRODataset(torch.utils.data.Dataset):
    def __init__(self, data_type, windowSize, color, transform=None):
        blur_path = 'data/blur'
        sharp_path = 'data/sharp'
        self.windowSize = windowSize
        self.color = color
        if data_type != 'train'  and data_type != 'test':
            raise ValueError('Invalid data type')
        self.data_type = data_type
        self.X_train = sorted(glob.glob(blur_path + '_train/*.png'))
        self.y_train = sorted(glob.glob(sharp_path + '_train/*.png'))
        self.X_test = sorted(glob.glob(blur_path + '_test/*.png'))
        self.y_test = sorted(glob.glob(sharp_path + '_test/*.png'))

    def __len__(self):
        if self.data_type == 'train':
            return len(self.X_train)
        if self.data_type == 'test':
            return len(self.X_test)

    def __getitem__(self, idx):
        if self.data_type == 'train':
            # if self.X_train[idx][12:-7] == self.y_train[idx][14:-9]:
            imgX = np.array(cv2.imread(self.X_train[idx]))
            imgy = np.array(cv2.imread(self.y_train[idx]))
            a = random.randint(0,imgX.shape[0]-self.windowSize-2)
            # print('Random call hua hai')
            b = random.randint(0,imgX.shape[1]-self.windowSize-2)
            imgX = imgX[a:a+self.windowSize,b:b+self.windowSize]
            imgy = imgy[a:a+self.windowSize,b:b+self.windowSize]
            X = np.array(imgX).reshape(3,self.windowSize,self.windowSize)
            y = np.array(imgy).reshape(3,self.windowSize,self.windowSize)

            # else:
            #     raise ValueError
            if not(self.color):
                X_temp = torch.from_numpy(X).float()
                X_temp = TF.to_tensor(TF.to_grayscale(TF.to_pil_image(X_temp))) * 255.0
                y_temp = torch.from_numpy(y).float()
                y_temp = TF.to_tensor(TF.to_grayscale(TF.to_pil_image(y_temp))) * 255.0
                return X_temp.float(),y_temp.float()

            else:
                return torch.from_numpy(X).float(), torch.from_numpy(y).float()

        if self.data_type == 'test':
            # if self.X_test[idx][12:-7] == self.y_test[idx][14:-9]:
            X = np.array(cv2.imread(self.X_test[idx]))
            X = X.reshape(3,X.shape[0],X.shape[1])
            y = np.array(cv2.imread(self.y_test[idx]))
            y = y.reshape(3,y.shape[0],y.shape[1])# Changes made in the above lines
            # else:
            #     raise ValueError

            return torch.from_numpy(X).float(), torch.from_numpy(y).float()
