
from src import *;



import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-quickie",type=int,default = 0);
parser.add_argument("-w",type=float,default=1.0);
parser.add_argument("-reset",type=int,default=0);

opt = parser.parse_args()

prepare_folders(opt.reset);

cnn, test_loader = train_mnist_cnn(quickie=quickie)
train_data = get_train_data(opt.quickie);

res = []

for class_emb_dim in [None, 3,10]:
	for schedType in ["sigmoid","linear","quad"]:
		diffModel, schedule = train_diff(train_data=train_data, schedType=schedType,
							   class_emb_dim=class_emb_dim, w=opt.w)

		fid_score, is_score = evaluate_diff_model(diffModel, cnn, test_loader, opt.w, schedule, numFakeIters=50,batch_size=100)
		d = {"w":opt.w,"class_emb_dim":class_emb_dim,"schedType":schedType,"fid":fid_score,"is_score":is_score}
		res.append(d)
		dumpRes(res);