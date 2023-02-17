
from src import *;



import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-quickie",type=int,default = 0);
parser.add_argument("-w",type=float,default=1.0);
parser.add_argument("-reset",type=int,default=0);
parser.add_argument("-version",type=int,default=0);
parser.add_argument("-dsName",type=str,default="MNIST");

opt = parser.parse_args()

prepare_folders(opt.reset);

cnn, train_loader, test_loader = train_cnn(quickie=opt.quickie, dsName = opt.dsName, epochs = (5 if opt.dsName == "MNIST" else 20));


if opt.w > 0:
	embedOptions = [3,"oneHot", 10]
else:
	embedOptions = [None]

while True:
	for class_emb_dim in embedOptions[::(-1)**(opt.version%2)]:
		for schedType in ["sigmoid","linear","quad"]:
			diffModel, schedule = train_diff(cnn, test_loader, train_loader=train_loader, schedType=schedType,quickie=opt.quickie,dsName=opt.dsName,
								   version=opt.version,epochs= (50 if not opt.quickie else 3),class_emb_dim=class_emb_dim, w=opt.w)

	opt.version *= 10