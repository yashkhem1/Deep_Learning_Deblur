import os
import argparse
from pprint import pprint

class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.opt = None

    def _initial(self):
        # ===============================================================
        #                     General options
        # ===============================================================
        self.parser.add_argument('--model_type' , type='str', default='SRN_Deblur', help='Select Model for deblurring')
        self.parser.add_argument('--lr', type=float,  default=0.0001 , help='Learning Rate for the model')
        self.parser.add_argument('--beta1', type=int, default=0.9, help='Value of Beta1 for adam optimizer')
        self.parser.add_argument('--train_batch_size', type=int, default=16, help='Training Batch Size')
        self.parser.add_argument('--test_batch_size', type=int, default=4, help='Testing Batch Size')
        self.parser.add_argument('--input_nc', type=int, default=3, help='Input Number of Channels ')
        self.parser.add_argument('--output_nc', type=int, default=3, help='Output Number of Channels ')
        self.parser.add_argument('--ngf', type=int, default=32, help='Number of Intermediate channels')
        self.parser.add_argument('--padding_type', type=str, default='replicate', help='Type of padding for convolution')

        # ===============================================================
        #                    SRN-DeblurNet options
        # ===============================================================
        self.parser.add_argument('--scale', type=float, default=0.5, help='Scale Value for SRN-DeblunNet')
        self.parser.add_argument('--n_levels', type=int, default=3, help='Number of Training levels for SRN-DeblunNet')
        self.parser.add_argument('--color', type=int, default=1, help='Set whether training on color images(1) or greyscale image(0)')
        self.parser.add_argument('--scale', type=float, default=0.5, help='Scale Value for SRN-DeblunNet')
        self.parser.add_argument('--train', type=int, default=1, help='Set training or test')



    def _print(self):
        print("\n==================Options=================")
        pprint(vars(self.opt), indent=4)
        print("==========================================\n")

    def parse(self):
        self._initial()
        self.opt = self.parser.parse_args()
        self._print()
        return self.opt
#
#
# class cgOptimizer():
#     def __init__(self, input_nc=None, output_nc=None, ngf=None, norm=None, no_dropout=None, n_blocks=None,
#                  padding_type=None, ndf=None, n_layers_D = None, pool_size = None, lr = None, beta1 = None, lambda_A = None, lambda_B = None, pool=None):
#         self.input_nc = input_nc
#         self.output_nc = output_nc
#         self.ngf = ngf
#         self.norm = norm
#         self.no_dropout = no_dropout
#         self.n_blocks = n_blocks
#         self.padding_type = padding_type
#         self.ndf = ndf
#         self.n_layers_D = n_layers_D
#         self.pool_size = pool_size
#         self.lr=lr
#         self.beta1 = beta1
#         self.lambda_A = lambda_A
#         self.lambda_B = lambda_B
#         self.pool = pool
#
# class p2pOptimizer():
#     def __init__(self, input_nc=None, output_nc=None, num_downs=None, ngf=None, norm_layer=None, use_dropout=None, ndf=None, n_layers_D=None, lr=None, beta1=None, lambda_L1=None, n_blocks=None, padding_type=None):
#         self.input_nc = input_nc
#         self.output_nc = output_nc
#         self.ngf = ngf
#         self.num_downs=num_downs
#         self.norm_layer=norm_layer
#         self.use_dropout=use_dropout
#         self.ndf=ndf
#         self.n_layers_D=n_layers_D
#         self.lr=lr
#         self.beta1=beta1
#         self.lambda_L1=lambda_L1
#         self.n_blocks=n_blocks
#         self.padding_type=padding_type