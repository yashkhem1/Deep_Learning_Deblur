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
        self.parser.add_argument('--model_type' , type=str, default='SRN_Deblur', help='Select Model for deblurring')
        self.parser.add_argument('--lr', type=float,  default=0.0001 , help='Learning Rate for the model')
        self.parser.add_argument('--epochs', type=float, default=2000, help='Number of epochs in training')
        self.parser.add_argument('--beta1', type=int, default=0.9, help='Value of Beta1 for adam optimizer')
        self.parser.add_argument('--train_batch_size', type=int, default=10, help='Training Batch Size')
        self.parser.add_argument('--test_batch_size', type=int, default=1, help='Testing Batch Size')
        self.parser.add_argument('--input_nc', type=int, default=3, help='Input Number of Channels ')
        self.parser.add_argument('--output_nc', type=int, default=3, help='Output Number of Channels ')
        self.parser.add_argument('--ngf', type=int, default=32, help='Number of Intermediate channels')
        self.parser.add_argument('--padding_type', type=str, default='replicate', help='Type of padding for convolution')
        self.parser.add_argument('--windowSize', type=int, default=256, help='Crop Window Size')
        self.parser.add_argument('--gpuID', type=int, default=0, help='default ID of gpu')

        # ===============================================================
        #                    SRN-DeblurNet options
        # ===============================================================
        self.parser.add_argument('--scale', type=float, default=0.5, help='Scale Value for SRN-DeblunNet')
        self.parser.add_argument('--n_levels', type=int, default=3, help='Number of Training levels for SRN-DeblunNet')
        self.parser.add_argument('--color', type=int, default=0, help='Set whether training on color images(1) or greyscale image(0)')
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