
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
	plt.savefig(getOutPath("img") + "imageGrid_"+suffix+".png")
	plt.close()