import torch
from code.cnn_utils import EncoderResblock,DecoderResblock,SRN_block
from code.init_model import init_weights
import itertools
from code.loss_utils import multi_scale_loss,resize2d
import torchvision.transforms.functional as transforms

torch.set_default_tensor_type('torch.FloatTensor')



def polynomial_lr(base_lr,iter,max_iter,power):
    return base_lr*((1-iter/max_iter)**power)

class SRN_Deblurnet():

    def __init__(self,opt):
        super(SRN_Deblurnet,self).__init__()
        self.device = torch.device("cuda:"+str(opt.gpuID) if torch.cuda.is_available() else "cpu")
        self.opt = opt
        self.scale = opt.scale
        self.n_levels = opt.n_levels
        self.color = opt.color
        self.epochs = opt.epochs
        if self.color:
            self.input_nc = 6
            self.output_nc = 3
        else:
            self.input_nc = 2
            self.output_nc = 1

        self.SRN_block = SRN_block(self.input_nc,self.output_nc,opt.ngf,opt.padding_type).to(self.device)
        # self.testSRN_block = SRN_block(6,3,opt.ngf,opt.padding_type).to(device)
        self.SRN_block.apply(init_weights)
        print(self.SRN_block)
        print("Parameters ", len(list(self.SRN_block.parameters())))
        # self.multi_scale_loss = multi_scale_loss

        self.base_lr = opt.lr
        self.optimizer = torch.optim.Adam(self.SRN_block.parameters(), lr = opt.lr, betas=[opt.beta1,0.999])

    def get_input(self,inputX,inputY):
        self.inputX = inputX.to(self.device)
        self.inputY = inputY.to(self.device)
        # if (not(self.opt.color)):
        #     self.inputX = transforms.to_tensor(transforms.to_grayscale(transforms.to_pil_image(self.inputX)))
        #     self.inputY = transforms.to_tensor(transforms.to_grayscale(transforms.to_pil_image(self.inputY)))


    def forward(self):
        # if self.opt.color:
        #     _, _, h, w = self.inputX.shape
        # else:
        #     _, h, w = self.inputX.shape
        _ , _, h, w = self.inputX.shape
        self.pred_list=[]
        inp_pred = self.inputX
        for  i in range(self.n_levels):
            scale = self.scale**(self.n_levels-i-1)
            hi = int(round(h*scale))
            wi = int(round(w*scale))
            inp_blur = resize2d(self.inputX,(hi,wi))
            inp_pred = resize2d(inp_pred,(hi,wi)).detach()
            inp_all = torch.cat([inp_blur,inp_pred],1)  ##Concatenating along the color channels
            inp_pred = self.SRN_block(inp_all).to(self.device)
            self.pred_list.append(inp_pred)
            del inp_blur,inp_all

    def forward_get(self,input):   ## Specifically for testing the model
        if torch.cuda.is_available():
            input = input.cuda()
        n, c, h, w = input.shape   ##For grayscale images c will be equal to 1
        # pred_list = []
        # if not(self.opt.color):
        #     input_grayscale = []
        #     for i in range(input.shape[0]):
        #         input_grayscale.append(input[i][0, :, :].unsqueeze(0).tolist())
        #         input_grayscale.append(input[i][1, :, :].unsqueeze(0).tolist())
        #         input_grayscale.append(input[i][2, :, :].unsqueeze(0).tolist())
        #         if torch.cuda.is_available():
        #             input = torch.tensor(input_grayscale).cuda() #.to(self.device)
        #         else:
        #             input = torch.tensor(input_grayscale)
                # input = torch.tensor(input_grayscale)
        # if torch.cuda.is_available():
        #     inp_pred = input.cuda()
        # else:
        inp_pred = input
        for i in range(self.n_levels):
            scale = self.scale ** (self.n_levels - i - 1)
            hi = int(round(h * scale))
            wi = int(round(w * scale))
            inp_blur = resize2d(input, (hi, wi))
            inp_pred = resize2d(inp_pred, (hi, wi)).detach()
            inp_all = torch.cat([inp_blur, inp_pred], 1)  ##Concatenating along the color channels
            inp_pred = self.SRN_block(inp_all) #.to(self.device)
            del inp_blur , inp_all
            # pred_list.append(inp_pred)

        return inp_pred


    def backward(self):
        # print("inputY ki shape" , self.inputY.shape)
        self.ms_loss = multi_scale_loss(self.inputY,self.pred_list,self.n_levels)
        self.ms_loss.backward()
        del self.pred_list, self.inputX, self.inputY

    def change_lr(self,iter):
        lr_new = polynomial_lr(self.base_lr,iter,self.epochs,0.3)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr_new
        self.opt.lr = lr_new

    def set_requires_grad(self, nets, requires_grad=False):

        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def optimize(self):
        self.forward()
        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()















