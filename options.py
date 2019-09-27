import os
import argparse

class Optimizer():
    def __init__(self, lr=None, mu=None, beta1=None, weight_decay=None,
                 lambda_L1=None, n_epochs=None, scheduler=None, batch_size=None):
        self.lr = lr
        self.mu = mu
        self.beta1 = beta1
        self.w_decay = weight_decay
        self.lambda_L1 = lambda_L1
        self.epochs = n_epochs
        self.scheduler = scheduler
        self.batch_size = batch_size

class cgOptimizer():
    def __init__(self, input_nc=None, output_nc=None, ngf=None, norm=None, no_dropout=None, n_blocks=None,
                 padding_type=None, ndf=None, n_layers_D = None, pool_size = None, lr = None, beta1 = None, lambda_A = None, lambda_B = None, pool=None):
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.norm = norm
        self.no_dropout = no_dropout
        self.n_blocks = n_blocks
        self.padding_type = padding_type
        self.ndf = ndf
        self.n_layers_D = n_layers_D
        self.pool_size = pool_size
        self.lr=lr
        self.beta1 = beta1
        self.lambda_A = lambda_A
        self.lambda_B = lambda_B
        self.pool = pool

class p2pOptimizer():
    def __init__(self, input_nc=None, output_nc=None, num_downs=None, ngf=None, norm_layer=None, use_dropout=None, ndf=None, n_layers_D=None, lr=None, beta1=None, lambda_L1=None, n_blocks=None, padding_type=None):
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.num_downs=num_downs
        self.norm_layer=norm_layer
        self.use_dropout=use_dropout
        self.ndf=ndf
        self.n_layers_D=n_layers_D
        self.lr=lr
        self.beta1=beta1
        self.lambda_L1=lambda_L1
        self.n_blocks=n_blocks
        self.padding_type=padding_type