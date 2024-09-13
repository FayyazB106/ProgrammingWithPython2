import argparse
import json
import utils

parser = argparse.ArgumentParser(description = "Predicting using neural network")

parser.add_argument('--top_k', dest = "top_k", action = "store", type = int, default = 5)
parser.add_argument('--arch', dest = "arch", action = "store", type = str, default = "vgg11")
parser.add_argument('--checkpoint', dest = "checkpoint", action = "store", default = "checkpoint.pth")
parser.add_argument('--img_path', dest = "img_path", action = "store", default = "flowers/test/16/image_06673.jpg")

args = parser.parse_args()
top_k = args.top_k
architecture = args.arch
checkpoint_path = args.checkpoint
image_path = args.img_path

model  = utils.load_checkpoint(checkpoint_path)

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

prob, classes = utils.predict(image_path, model, top_k, cat_to_name)

for i in range(len(classes)):
    print("Flower name:{}, Probability:{}".format(classes[i],probs[0][i].tolist()))