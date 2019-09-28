from code.train import train
from opt import Options

if __name__=="__main__":
    opt = Options().parse()
    train(opt,opt.model_type)