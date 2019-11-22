import torch
import torch.nn as nn
import sys

from code.SRN_DeblurNet import SRN_Deblurnet
from torch.optim import lr_scheduler
import time
import copy
from torch.autograd import Variable
from code.utils import normalization, denormalize, scale_down, scale_up
from code.data_loader import GOPRODataset
import cv2
import numpy as np
import os
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import pickle

# from torchvision.transforms import ToPILImage

torch.set_default_tensor_type('torch.FloatTensor')



# if torch.cuda.is_available():
# 	torch.set_default_tensor_type('torch.cuda.FloatTensor')
# else:
# 	torch.set_default_tensor_type('torch.FloatTensor')

def train(opt, model_name):
	device = torch.device("cuda:"+str(opt.gpuID) if torch.cuda.is_available() else "cpu")

	train_set = GOPRODataset('train', opt.windowSize, opt.color)
	print('Loaded training set')
	test_set = GOPRODataset('test', opt.windowSize, opt.color)
	print('Loaded val set')

	train_loader = torch.utils.data.DataLoader(train_set,batch_size=opt.train_batch_size,shuffle=True, num_workers=0)
	test_loader = torch.utils.data.DataLoader(test_set,batch_size=opt.test_batch_size,shuffle=True, num_workers=0)

	train_length = len(train_set)
	test_length = len(test_set)
	print( "Training Data Size : ", train_length)
	print("Test Data Size : ", test_length)


	if model_name == 'SRN_Deblur':  # This is for Khem's version of Pix2Pix
		model = SRN_Deblurnet(opt)
		print_freq = 10
		train_iter = iter(train_loader)
		test_iter = iter(test_loader)
		fixed_X , fixed_Y = test_iter.next()
		fixed_X = scale_down(fixed_X).to(device)
		fixed_Y = scale_down(fixed_Y).to(device)
		loss_multi_scale = []

		num_batches = len(train_iter)
		for epoch in range(2000):

			model.change_lr(epoch)
			train_iter = iter(train_loader)
			since = time.time()
			print("Epoch ", epoch ," entering ")
			for batch in range(num_batches):
				print("Epoch ", epoch ,"Batch ", batch, " running with learning rate ", model.opt.lr)
				inputX,inputY = train_iter.next()
				inputX = scale_down(inputX).to(device)
				inputY = scale_down(inputY).to(device)
				# if batch==1:
				# 	cv2.imwrite(os.path.join('srn_results/input_train',
				# 							 'blur_{}_{}_{}.png'.format(batch, batch, epoch)),
				# 				np.array(scale_up(inputX[1]).cpu().detach()).reshape(inputX[1].shape[1],
				# 																	inputX[1].shape[2], 3))
				model.get_input(inputX,inputY)
				model.optimize()
				del inputX,inputY
				torch.cuda.empty_cache()
				# print("Dx Loss : {:.6f} Dy Loss: {:.6f} Generator Loss: {:.6f} ".format(model.dx_loss, model.dy_loss, model.gen_loss))
				print("Model Multi Scale Loss " , float(model.ms_loss))



			if (epoch+1)%1 == 0:
				torch.set_grad_enabled(False)
				sharp = model.forward_get(fixed_X)
				if not(opt.color):
					numImages = int(sharp.shape[0]/3)
				else:
					numImages = int(sharp.shape[0])
				for j in range(numImages):
					if not(opt.color):
						sharp_image = torch.cat([sharp[j*3],sharp[j*3+1], sharp[j*3+2]],0)
					else:
						sharp_image = sharp[j]
					cv2.imwrite( os.path.join('srn_results/pred_sharp',
						'sharp_{}_{}_{}.png'.format(batch, j, epoch)),
					  np.array(scale_up(sharp_image).cpu().detach()).reshape(sharp_image.shape[1],sharp_image.shape[2],3))

					cv2.imwrite( os.path.join('srn_results/inputs',
						'blur_{}_{}_{}.png'.format(batch, j, epoch)),
						np.array(scale_up(fixed_X[j]).cpu().detach()).reshape(fixed_X[j].shape[1],fixed_X[j].shape[2], 3))
					cv2.imwrite( os.path.join('srn_results/inputs',
						'ground_sharp_{}_{}_{}.png'.format(batch, j, epoch)),
						np.array(scale_up(fixed_Y[j]).cpu().detach()).reshape(fixed_Y[j].shape[1],fixed_Y[j].shape[2], 3))


				torch.set_grad_enabled(True)

			print("Time to finish epoch ", time.time()-since)

			torch.save(model, 'SRNmodel/best_model_new.pt')
			loss_multi_scale.append(float(model.ms_loss.detach()))
			with open('SRNloss/loss_ms_new.pk', 'wb') as f:
				pickle.dump(loss_multi_scale, f)


# train(opt, 'P2P')
