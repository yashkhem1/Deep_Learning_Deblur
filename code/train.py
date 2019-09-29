import torch
import torch.nn as nn
import sys
from code.pix2pix import Pix2Pix
from code.cycleGAN import cycleGan
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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# if torch.cuda.is_available():
# 	torch.set_default_tensor_type('torch.cuda.FloatTensor')
# else:
# 	torch.set_default_tensor_type('torch.FloatTensor')

def train(opt, model_name):

	train_set = GOPRODataset('train', opt.windowSize)
	print('Loaded training set')
	test_set = GOPRODataset('test', opt.windowSize)
	print('Loaded val set')

	train_loader = torch.utils.data.DataLoader(train_set,batch_size=opt.train_batch_size,shuffle=True, num_workers=0)
	test_loader = torch.utils.data.DataLoader(test_set,batch_size=opt.test_batch_size,shuffle=True, num_workers=0)

	train_length = len(train_set)
	test_length = len(test_set)
	print( "Training Data Size : ", train_length)
	print("Test Data Size : ", test_length)

	# dataloader = {x: torch.utils.data.DataLoader(
	# 	dataset[x], batch_size=opt.batch_size, shuffle=True, num_workers=0) for x in range(2)}
	#
	# cg_train_loader = torch.utils.data.DataLoader(cg_dataset[0], batch_size=2, shuffle=True, num_workers=0)
	# cg_val_loader = torch.utils.data.DataLoader(cg_dataset[1], batch_size=4, shuffle=True, num_workers=0)
	#
	# p2p_train_loader = torch.utils.data.DataLoader(p2p_dataset[0], batch_size=4, shuffle=True, num_workers=0)
	# p2p_val_loader = torch.utils.data.DataLoader(p2p_dataset[1], batch_size=4, shuffle=True, num_workers=0)
	#
	# dataset_size = {x: len(dataset[x]) for x in range(2)}

	if model_name == 'CycleGAN':
		model = cycleGan(opt)
		print_freq = 10
		train_iter = iter(train_loader)
		test_iter = iter(test_loader)
		fixed_X , fixed_Y = test_iter.next()
		fixed_X = normalization(fixed_X).to(device)
		fixed_Y = normalization(fixed_Y).to(device)
		loss_Gl = []
		loss_DXl = []
		loss_DYl = []


		num_batches = len(train_iter)
		for epoch in range(200):
			if epoch == 35:
				model.change_lr(model.opt.lr/2)

			if epoch == 80:
				model.change_lr(model.opt.lr/2)

			if epoch == 130:
				model.change_lr(model.opt.lr/2)


			since = time.time()
			print("Epoch ", epoch ," entering ")
			for batch in range(num_batches):
				print("Epoch ", epoch ,"Batch ", batch, " running with learning rate ", model.opt.lr)
				inputX,inputY = train_iter.next()
				inputX = normalization(inputX).to(device)
				inputY = normalization(inputY).to(device)
				model.get_input(inputX,inputY)
				model.optimize()
				# print("Dx Loss : {:.6f} Dy Loss: {:.6f} Generator Loss: {:.6f} ".format(model.dx_loss, model.dy_loss, model.gen_loss))
				print("Model dx loss " , float(model.loss_D_X), "Model dy loss", float(model.loss_D_Y), "model_gen_loss", float(model.loss_G))



			if (epoch+1)%10 == 0:
				# torch.set_grad_enabled(False)
				sharp = model.G_XtoY.forward(fixed_X)
				for j in range(sharp.size()[0]):
					if opt.n_blocks==6:
						cv2.imwrite( os.path.join('../cgresults/pred_masks',
							'mask_{}_{}_{}.png'.format(batch, j, epoch)),
						  np.array(denormalize(sharp[j]).cpu().detach()).reshape(sharp[j].shape[1],sharp[j].shape[2],3))
						if epoch == 9:
							cv2.imwrite( os.path.join('../cgresults/inputs',
								'input_{}_{}_{}.png'.format(batch, j, epoch)),
							  np.array(denormalize(fixed_X[j]).cpu().detach()).reshape(fixed_X[j].shape[1],fixed_X[j].shape[2], 3))
					else:
						cv2.imwrite( os.path.join('../cgresults/r-9-pred_masks',
									'mask_{}_{}_{}.png'.format(batch, j, epoch)),
								  np.array(denormalize(sharp[j]).cpu().detach()).reshape(sharp[j].shape[1],sharp[j].shape[2],3))
						if epoch == 9:
							cv2.imwrite( os.path.join('../cgresults/r-9-inputs',
								'input_{}_{}_{}.png'.format(batch, j, epoch)),
							  np.array(denormalize(fixed_X[j]).cpu().detach()).reshape(fixed_X[j].shape[1],fixed_X[j].shape[2], 3))

				# torch.set_grad_enabled(True)

			print("Time to finish epoch ", time.time()-since)

			torch.save(model, '../CGmodel/best_model5.pt')
			loss_Gl.append(float(model.loss_G))
			loss_DXl.append(float(model.loss_D_X))
			loss_DYl.append(float(model.loss_D_Y))
			with open('../CGloss/lossG5.pk', 'wb') as f:
				pickle.dump(loss_Gl, f)
			with open('../CGloss/lossD_X5.pk', 'wb') as f:
				pickle.dump(loss_DXl, f)
			with open('../CGloss/lossd_Y5.pk', 'wb') as f:
				pickle.dump(loss_DYl, f)


	elif model_name == 'P2P':  # This is for Khem's version of Pix2Pix
		model = Pix2Pix(opt)
		print_freq = 10
		train_iter = iter(train_loader)
		test_iter = iter(test_loader)
		fixed_X , fixed_Y = test_iter.next()
		fixed_X = normalization(fixed_X).to(device)
		fixed_Y = normalization(fixed_Y).to(device)
		loss_Gl = []
		loss_Dl = []


		num_batches = len(train_iter)
		for epoch in range(3000):

			if epoch == 299:
				model.change_lr(model.opt.lr/2)

			if epoch == 499:
				model.change_lr(model.opt.lr/2)

			since = time.time()
			print("Epoch ", epoch ," entering ")
			for batch in range(num_batches):
				print("Epoch ", epoch ,"Batch ", batch, " running with learning rate ", model.opt.lr)
				inputX,inputY = train_iter.next()
				inputX = normalization(inputX).to(device)
				inputY = normalization(inputY).to(device)
				model.get_input(inputX,inputY)
				model.optimize()
				# print("Dx Loss : {:.6f} Dy Loss: {:.6f} Generator Loss: {:.6f} ".format(model.dx_loss, model.dy_loss, model.gen_loss))
				print("Model D Loss " , float(model.loss_D), "Model G loss", float(model.loss_G))



			if (epoch+1)%10 == 0:
				# torch.set_grad_enabled(False)
				sharp = model.G.forward(fixed_X)
				for j in range(sharp.size()[0]):
					cv2.imwrite( os.path.join('../p2presults/pred_masks',
						'mask_{}_{}_{}.png'.format(batch, j, epoch)),
					  np.array(denormalize(sharp[j]).cpu().detach()).reshape(sharp[j].shape[1],sharp[j].shape[2],3))
					if epoch == 9:
						cv2.imwrite( os.path.join('../p2presults/inputs',
							'input_{}_{}_{}.png'.format(batch, j, epoch)),
						  np.array(denormalize(fixed_X[j]).cpu().detach()).reshape(fixed_X[j].shape[1],fixed_X[j].shape[2], 3))
						cv2.imwrite( os.path.join('../p2presults/inputs',
							'ground_depth_{}_{}_{}.png'.format(batch, j, epoch)),
						  np.array(denormalize(fixed_Y[j]).cpu().detach()).reshape(fixed_Y[j].shape[1],fixed_Y[j].shape[2], 3))


				# torch.set_grad_enabled(True)

			print("Time to finish epoch ", time.time()-since)

			torch.save(model, '../P2Pmodel/best_model8.pt')
			loss_Gl.append(float(model.loss_G))
			loss_Dl.append(float(model.loss_D))
			with open('../P2Ploss/lossG8.pk', 'wb') as f:
				pickle.dump(loss_Gl, f)
			with open('../P2Ploss/lossD8.pk', 'wb') as f:
				pickle.dump(loss_Dl, f)


	elif model_name == 'SRN_Deblur':  # This is for Khem's version of Pix2Pix
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
				for j in range(sharp.size()[0]):
					cv2.imwrite( os.path.join('srn_results/pred_sharp',
						'sharp_{}_{}_{}.png'.format(batch, j, epoch)),
					  np.array(scale_up(sharp[j]).cpu().detach()).reshape(sharp[j].shape[1],sharp[j].shape[2],3))

					cv2.imwrite( os.path.join('srn_results/inputs',
						'blur_{}_{}_{}.png'.format(batch, j, epoch)),
						np.array(scale_up(fixed_X[j]).cpu().detach()).reshape(fixed_X[j].shape[1],fixed_X[j].shape[2], 3))
					cv2.imwrite( os.path.join('srn_results/inputs',
						'ground_sharp_{}_{}_{}.png'.format(batch, j, epoch)),
						np.array(scale_up(fixed_Y[j]).cpu().detach()).reshape(fixed_Y[j].shape[1],fixed_Y[j].shape[2], 3))


				torch.set_grad_enabled(True)

			print("Time to finish epoch ", time.time()-since)

			torch.save(model, 'SRNmodel/best_model.pt')
			loss_multi_scale.append(float(model.ms_loss.detach()))
			with open('SRNloss/loss_ms.pk', 'wb') as f:
				pickle.dump(loss_multi_scale, f)


# train(opt, 'P2P')
