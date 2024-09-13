import argparse
import utils

parser = argparse.ArgumentParser(description = "Training neural network")

parser.add_argument('--arch', dest = "arch", action = "store", type = str, default = "vgg11")
parser.add_argument('--learning_rate', dest = "learning_rate", action = "store", type = float, default = 0.001)
parser.add_argument('--dropout', dest = "dropout", action = "store", type = float, default = 0.2)
parser.add_argument('--epochs', dest = "epochs", action = "store", type = int, default = 5)
parser.add_argument('--gpu', dest = "gpu",type = bool, action = "store", default = "True")
parser.add_argument('--save_dir', dest = "save_dir", action = "store", default = "checkpoint.pth")

args = parser.parse_args()

architecture = args.arch
learning_rate = args.learning_rate
dropout = args.dropout
epochs = args.epochs
gpu = args.gpu
path = args.save_dir

print("Architecture: {}".format(architecture))
print("Learning Rate: {}".format(learning_rate))
print("Dropout: {}".format(dropout))
print("Number of Epochs: {}".format(epochs))
print("GPU Enabled: {}".format(gpu))
print("Save Directory: {}".format(path))

trainloader, validloader, testloader, train_data = utils.sort_data()  
model, criterion, optimizer, device = utils.create_model(architecture, learning_rate, dropout, gpu, train_data)
model, optimizer  = utils.train_model(epochs, model, criterion, optimizer, device, trainloader, validloader)
utils.test_model(testloader, model, device)
utils.save_checkpoint(path, dropout, architecture, learning_rate, optimizer, model)
