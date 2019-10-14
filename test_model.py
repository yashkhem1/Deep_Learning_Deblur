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


testImage =  np.array(cv2.imread(testImage))
cv2.imwrite( './testInput.jpeg',np.array(testImage).reshape(testImage.shape[0], testImage.shape[1], 3))
testImage = torch.from_numpy(np.array(testImage).reshape(1,3,testImage.shape[0],testImage.shape[1])).float()
testImage = scale_down(testImage).to(device)
print(testImage)
output=None
if modelType == 'p':
    if torch.cuda.is_available():
        model = torch.load("./P2Pmodel/" + bestModel)
        gen = model.G.to(device)
    else:
        model = torch.load("./P2Pmodel/" + bestModel, map_location='cpu')
        gen = model.G
    output = gen.forward(testImage)
elif modelType == 'c':
    if torch.cuda.is_available():
        model = torch.load("./CGmodel/" + bestModel)
        gen = model.G_XtoY.to(device)
    else:
        model = torch.load("./CGmodel/" + bestModel, map_location='cpu')
        gen = model.G_XtoY
    output = gen.forward(testImage)

elif modelType=='srn':
    # inp_pred = testImage
    if torch.cuda.is_available():
        model = torch.load("./SRNmodel/" + bestModel, map_location='cuda:0')
        torch.set_grad_enabled(False)
        output = model.forward_get(testImage)
        # n, c, h, w = testImage.shape
        # # pred_list = []
        #
        # for i in range(model.n_levels):
        #     scale = model.scale ** (model.n_levels - i - 1)
        #     hi = int(round(h * scale))
        #     wi = int(round(w * scale))
        #     inp_blur = resize2d(testImage, (hi, wi))
        #     inp_pred = resize2d(inp_pred, (hi, wi)).detach()
        #     inp_all = torch.cat([inp_blur, inp_pred], 1)  ##Concatenating along the color channels
        #     inp_pred = model.SRN_block(inp_all).to(device)
        #     del inp_blur, inp_all
            # pred_list.append(inp_pred)


        torch.set_grad_enabled(True)
    else:
        model = torch.load("./SRNmodel/" + bestModel, map_location='cpu')
        torch.set_grad_enabled(False)
        output=model.forward_get(testImage)
        # n, c, h, w = testImage.shape
        # # pred_list = []
        #
        # for i in range(model.n_levels):
        #     scale = model.scale ** (model.n_levels - i - 1)
        #     hi = int(round(h * scale))
        #     wi = int(round(w * scale))
        #     inp_blur = resize2d(testImage, (hi, wi))
        #     inp_pred = resize2d(inp_pred, (hi, wi)).detach()
        #     inp_all = torch.cat([inp_blur, inp_pred], 1)  ##Concatenating along the color channels
        #     inp_pred = model.SRN_block(inp_all).to(device)
        #     del inp_blur, inp_all
        #     # pred_list.append(inp_pred)


        torch.set_grad_enabled(True)

    # output=inp_pred


if color:
    cv2.imwrite( './testOutput.jpeg',np.array(scale_up(output).cpu().detach()).reshape(testImage.shape[2], testImage.shape[3], 3))

else:
    sharp_image = torch.cat([output[0], output[1],  output[2]], 0)
    cv2.imwrite('./testOutput.jpeg',
                np.array(scale_up(sharp_image).cpu().detach()).reshape(testImage.shape[2], testImage.shape[3], 3))
