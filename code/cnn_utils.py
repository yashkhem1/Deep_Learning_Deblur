import torch
import torch.nn as nn
import functools
import torch.nn.functional as F


def get_norm_layer(norm_type="batch"):
    if(norm_type == "batch"):
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif (norm_type == "instance"):
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    else:
        norm_layer = None

    return norm_layer


class ResnetBlock(nn.Module):

    def __init__(self,dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock,self).__init__()
        model = []
        p = 0
        if padding_type == 'reflect':
            model += [nn.ReflectionPad2d(2)]
        elif padding_type == 'replicate':
            model += [nn.ReplicationPad2d(2)]
        else:
            p=2

        if norm_layer is not None:
             model += [nn.Conv2d(dim, dim, kernel_size=5, padding=p, stride=1), norm_layer(dim), nn.ReLU(True)]
        else:
            model += [nn.Conv2d(dim, dim, kernel_size=5, padding=p, stride=1),  nn.ReLU(True)]
        if use_dropout:
            model+= [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            model += [nn.ReflectionPad2d(2)]
        elif padding_type == 'replicate':
            model += [nn.ReplicationPad2d(2)]
        else:
            p=2

        if norm_layer is not None:
            model += [nn.Conv2d(dim, dim, kernel_size=5, padding=p, stride=1), norm_layer(dim)]

        else:
            model += [nn.Conv2d(dim, dim, kernel_size=5, padding=p, stride=1)]

        self.model = nn.Sequential(*model)

    def forward(self,x):
        # print("Resenet me shape" , x.shape)
        out = x + self.model(x)
        return out



##### For Scale-Recurrent Network ##########

class EncoderResblock(nn.Module):

    def __init__(self,input_nc, output_nc, half, padding_type='replicate', norm_layer=None):
        super(EncoderResblock,self).__init__()
        if half:
            model = [nn.ReflectionPad2d(2), nn.Conv2d(input_nc, output_nc, kernel_size=5, stride=2, padding=0),
                     nn.ReLU(True)]
        else:
            model = [nn.ReflectionPad2d(2), nn.Conv2d(input_nc, output_nc, kernel_size=5, stride=1, padding=0),
                     nn.ReLU(True)]

        for i in range(3):
            model+= [ResnetBlock(output_nc,padding_type,norm_layer,False,False)]

        self.model = nn.Sequential(*model)

    def forward(self,input):
        # print("Shaep of input ", input.shape)
        return self.model(input)


class DecoderResblock(nn.Module):
    def __init__(self, input_nc, output_nc, double,  padding_type='replicate', norm_layer=None):
        super(DecoderResblock, self).__init__()
        model=[]
        for i in range(3):
            model += [ResnetBlock(input_nc, padding_type, norm_layer, False, False)]

        if double:
            model += [nn.ConvTranspose2d(input_nc, output_nc, kernel_size=4, stride=2, padding=1),
                     nn.ReLU(True)]
        else:
            model += [nn.ConvTranspose2d(input_nc, output_nc, kernel_size=5, stride=1, padding=2),
                     nn.ReLU(True)]

        # for i in range(3):
        #     model+= [ResnetBlock(output_nc,padding_type,norm_layer,False,False)]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        # print("Shaep of input Decoder ", input.shape)
        return self.model(input)


class SRN_block(nn.Module):
    def __init__(self,input_nc,output_nc,ngf=32,padding_type='replicate'):
        super(SRN_block,self).__init__()
        # model = []
        self.erb1 = EncoderResblock(input_nc,ngf,False,padding_type)
        self.erb2 = EncoderResblock(ngf, ngf*2, True, padding_type)
        self.erb3 = EncoderResblock(ngf*2, ngf*4, True, padding_type)
        ## Insert LSTM here later
        self.drb3 = DecoderResblock(ngf*4,ngf*2,True,padding_type)
        self.drb2 = DecoderResblock(ngf*2, ngf, True, padding_type)
        model=[]
        for i in range(3):
            model += [ResnetBlock(ngf, padding_type,None, False, False)]
        model += [nn.ReflectionPad2d(2), nn.Conv2d(ngf, output_nc, kernel_size=5, stride=1, padding=0)] #Without RELU

        self.drb1 = nn.Sequential(*model)

    def forward(self,input):
        # return self.model(input)
        erb1x = self.erb1(input)
        erb2x = self.erb2(erb1x)
        erb3x = self.erb3(erb2x)
        drb3x = self.drb3(erb3x)
        drb3x = drb3x + erb2x
        drb2x = self.drb2(drb3x)
        drb2x = drb2x + erb1x
        out = self.drb1(drb2x)
        return out




