import argparse
from train import *

parser = argparse.ArgumentParser(description="Parsing Params")
parser.add_argument('--lr', type=float, default=.001, help="Set learning rate of opt")
parser.add_argument('--epoch', '-e', type=int, default=75, help="Set number of epochs")
parser.add_argument('--samples', '-s', type=int, default=2000, help="Set number of samples")
parser.add_argument('--batch', '-bs', type=int, default=4, help="Set batch size")
parser.add_argument('--data' 'd', type=str, help="Specify h5 path")

args = parser.parse_args()

lr = args.lr
epochs = args.epoch
samples = args.samples
batch_size = args.batch
h5 = args.data

train(lr=lr, epochs=epochs, samples=samples, batch_size=batch_size, h5_path=h5)