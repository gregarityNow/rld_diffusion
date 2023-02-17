
from .basis_funcs import *
from torchvision.utils import make_grid

from torchvision.utils import make_grid

# %matplotlib inline
import matplotlib.pyplot as plt

def show_images(images, suffix, nrow=10, labels=None):
	"""Show images

	Args:
		images (torch.Tensor): The batch of images
		nrow (int, optional): The number of images per row. Defaults to 8.
	"""
	images_3c = images.repeat(1, 3, 1, 1) if images.shape[1] != 3 else images
	images_3c = images_3c.double().clamp(0, 1)
	grid = make_grid(images_3c, nrow=nrow).permute((1, 2, 0))
	if labels is not None:
		plt.xticks(torch.arange(labels.shape[0]).numpy() * images.shape[3] * 1.08 + images.shape[3] / 2, labels=labels.numpy())

	plt.imshow(grid)
	imgOut = getOutPath("img") + "imageGrid_"+suffix+".png"
	plt.savefig(imgOut)
	print("saved image to",imgOut)
	plt.close()


def get_w_scores(df, noCifar = True):
	df = df[(df.epoch==50)]
	if noCifar:
		df = df[df.dsName != "CIFAR10"]

	print(df.groupby("w")[["fid", "is_score"]].mean().round(2).reset_index().to_latex(index=False))

# ws = df.groupby("w")[["fid", "is_score"]].mean().sort_values("is_score")
# plt.plot(ws.is_score, ws.fid)
# plt.ylabel("IS")
# plt.xlabel("FID")
# plt.title("Mean FID vs IS for MNIST images after 50 epochs")



def get_embed_scores(df, noCifar=True):
	df = df[(df.epoch == 50)]
	if noCifar:
		df = df[df.dsName != "CIFAR10"]
	df.class_emb_dim = df.class_emb_dim.fillna("No Guidance")
	print(df.groupby("class_emb_dim")[["fid", "is_score"]].mean().round(2).reset_index().to_latex(index=False))

def get_sched_scores(df, noCifar=True):
	df = df[(df.epoch == 50)]
	if noCifar:
		df = df[df.dsName != "CIFAR10"]
	df.class_emb_dim = df.class_emb_dim.fillna("No Guidance")
	print(df.groupby("schedType")[["fid", "is_score"]].mean().round(2).reset_index().to_latex(index=False))



def get_epoch_scores(df, noCifar=True):
	if noCifar:
		df = df[df.dsName != "CIFAR10"]
	df.class_emb_dim = df.class_emb_dim.fillna("No Guidance")
	print(df.groupby("epoch")[["fid", "is_score","loss"]].mean().round(3).reset_index().to_latex(index=False))

def get_mnist_vs_cifar_scores(df):
	df = df[(df.epoch == 50)]
	df.dsName = df.dsName.fillna("MNIST")
	print(df.groupby("dsName")[["fid", "is_score"]].mean().round(2).reset_index().to_latex(index=False))
