import argparse
from train import *
import pyk4a

parser = argparse.ArgumentParser(description="Parsing Params")
parser.add_argument('--lr', type=float, default=.001, help="Set learning rate of opt")
parser.add_argument('--epoch', '-e', type=int, default=1, help="Set number of epochs")
parser.add_argument('--batch', '-bs', type=int, default=1, help="Set batch size")
parser.add_argument('--data_path', '-dp', type=str, default='./', help="File path of our data")
parser.add_argument('--checkpoint_path', '-cp', type=str, default='./', help="File path of our pretrained model")

args = parser.parse_args()

lr = args.lr
epochs = args.epoch
batch_size = args.batch
data_path = args.data_path
checkpoint_path = args.checkpoint_path


train_with_local(data_path, checkpoint_path, lr=lr, epochs=epochs, batch_size=batch_size)