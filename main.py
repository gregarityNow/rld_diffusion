
from src import *;



import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-quickie",type=int,default = 0);
parser.add_argument("-w",type=float,default=1.0);
parser.add_argument("-reset",type=int,default=0);
parser.add_argument("-version",type=int,default=0);

opt = parser.parse_args()

prepare_folders(opt.reset);

cnn, test_loader = train_mnist_cnn(quickie=opt.quickie)
train_data = get_train_data(opt.quickie);


if opt.w > 0:
	embedOptions = [3,"oneHot", 10]
else:
	embedOptions = [None]

for class_emb_dim in embedOptions[::(-1)**(opt.version%2)]:
	for schedType in ["sigmoid","linear","quad"]:
		diffModel, schedule = train_diff(cnn, test_loader, train_data=train_data, schedType=schedType,quickie=opt.quickie,
							   version=opt.version,epochs= (50 if not opt.quickie else 3),class_emb_dim=class_emb_dim, w=opt.w)

