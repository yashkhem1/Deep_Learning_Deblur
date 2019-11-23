import torch
import sys
import cv2
from code.utils import normalization,denormalize,scale_down, scale_up
import numpy as np
from code.loss_utils import resize2d

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

bestModel = sys.argv[1]
testImage = sys.argv[2]
modelType = sys.argv[3]
color = int(sys.argv[4])
testGray = int(sys.argv[5])   #If we are testing on GrayScale images

if color:
    testImage = cv2.imread(testImage)
    if testImage.shape[0]!=720 or testImage.shape[1]!=1280:
        testImage = np.array(cv2.resize(testImage,(1280,720)))
        print('hello')
    else:
        testImage = np.array(testImage)
    cv2.imwrite( './testInput.jpeg',np.array(testImage).reshape(testImage.shape[0], testImage.shape[1], 3))
    testImage = torch.from_numpy(np.array(testImage).reshape(1,3,testImage.shape[0],testImage.shape[1])).float()
    testImage = scale_down(testImage).to(device)

else:
    if testGray:
        testImage = cv2.cvtColor(cv2.imread(testImage),cv2.COLOR_BGR2GRAY)
        if testImage.shape[0] != 720 or testImage.shape[1] != 1280:
            testImage = np.array(cv2.resize(testImage, (1280, 720)))
        else:
            testImage = np.array(testImage)
        cv2.imwrite('./testInput.jpeg', np.array(testImage).reshape(testImage.shape[0], testImage.shape[1], 1))
        testImage = torch.from_numpy(np.array(testImage).reshape(1, 1, testImage.shape[0], testImage.shape[1])).float()
        testImage = scale_down(testImage).to(device)

    else:
        input_grayscale = []
        testImage = cv2.imread(testImage)
        if testImage.shape[0] != 720 or testImage.shape[1] != 1280  :
            testImage = np.array(cv2.resize(testImage, (1280, 720)))
        else:
            testImage = np.array(testImage)
        cv2.imwrite('./testInput.jpeg', np.array(testImage).reshape(testImage.shape[0], testImage.shape[1], 3))
        testImage = torch.from_numpy(np.array(testImage).reshape(1, 3, testImage.shape[0], testImage.shape[1])).float()
        # print(testImage.shape)
        input_grayscale.append(testImage[0,0, :, :].unsqueeze(0).tolist())
        input_grayscale.append(testImage[0,1, :, :].unsqueeze(0).tolist())
        input_grayscale.append(testImage[0,2, :, :].unsqueeze(0).tolist())
        testImage = torch.tensor(input_grayscale).float()
        # print("Shape of Test Image ", testImage.shape)
        testImage = scale_down(testImage).to(device)


# print(testImage)
output=None

if modelType=='srn':
    # inp_pred = testImage
    if torch.cuda.is_available():
        model = torch.load("./SRNmodel/" + bestModel, map_location='cuda:0')
        torch.set_grad_enabled(False)
        output = model.forward_get(testImage)
        torch.set_grad_enabled(True)

    else:
        model = torch.load("./SRNmodel/" + bestModel, map_location='cpu')
        torch.set_grad_enabled(False)
        output=model.forward_get(testImage)
        torch.set_grad_enabled(True)

    # output=inp_pred


if color:
    cv2.imwrite( './testOutput.jpeg',np.array(scale_up(output).cpu().detach()).reshape(testImage.shape[2], testImage.shape[3], 3))

else:
    if not testGray:
        # print("Output Shape ", output.shape)
        sharp_image = torch.cat([output[0], output[1],  output[2]], 0)
        # print("Sharp image shape", sharp_image.shape)
        cv2.imwrite('./testOutput.jpeg',
                    np.array(scale_up(sharp_image).cpu().detach()).reshape(sharp_image.shape[1], sharp_image.shape[2], 3))
    else:
        cv2.imwrite('./testOutput.jpeg',np.array(scale_up(output).cpu().detach()).reshape(testImage.shape[2], testImage.shape[3], 1))
