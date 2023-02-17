from .basis_funcs import *;

from torch.utils.data import DataLoader
from datamaestro import prepare_dataset

class CNN(nn.Module):
	def __init__(self):
		super().__init__()
		self.net = nn.Sequential(
			nn.Conv2d(1, 64 ,(5 ,5), stride=1, padding=2),
			nn.ReLU(),
			nn.MaxPool2d((2 ,2)),
			nn.Conv2d(64, 32, (5 ,5), stride=1, padding=2),
			nn.ReLU(),
			nn.MaxPool2d((2 ,2)),
			nn.Conv2d(32, 16, (5 ,5), stride=1, padding=2),
			nn.ReLU(),
			nn.MaxPool2d((2 ,2)),
		)
		self.classif = nn.Linear(144, 10)
	def forward(self, x):
		# TODO
		bsize = x.size(0)
		emb = self.net(x)
		embFlat = emb.view(bsize, -1)
		out = self.classif(embFlat)
		# print("nose",out.shape)
		return out

	def get_act_and_class(self, x):
		with torch.no_grad():
			bsize = x.size(0)
			emb = self.net(x)
			act = emb.view(bsize, -1)
			logits = self.classif(act)
		return act.detach().cpu(), logits.detach().cpu()

from torchvision import transforms
import torchvision
from torch.nn import functional as F

def eval_model(net, loader):
	net.eval()

	acc, loss = 0, 0.
	c = 0
	for x, y in loader:
		c += len(x)

		with torch.no_grad():
			logits = net(x.cuda()).cpu()

		loss += F.cross_entropy(logits, y).item()
		acc += (logits.argmax(dim=1) == y).sum().item()

	return round(100 * acc / c, 2), round(loss / len(loader), 5)

batch_size = 100
def train_mnist_cnn(epochs = 5, quickie = False):


	mnist_pixels = torchvision.datasets.MNIST(train=True, download=True,root=dataLoc).data / 255
	mean = mnist_pixels.mean().item()
	std = mnist_pixels.std().item()

	print(f"Mean {mean} and Std {std}")
	mean = torch.tensor([mean])
	std = torch.tensor([std])

	transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize(mean, std)
	])

	mnist_train = torchvision.datasets.MNIST(train=True, transform=transform,download=True,root=dataLoc)
	mnist_test = torchvision.datasets.MNIST(train=False, transform=transform,download=True,root=dataLoc)

	if quickie:
		mnist_train = torch.utils.data.DataLoader(torch.utils.data.Subset(mnist_train,torch.arange(100)), batch_size=4,
													shuffle=True, num_workers=2)
		mnist_test = torch.utils.data.DataLoader(torch.utils.data.Subset(mnist_test, torch.arange(100)), batch_size=4,
												  shuffle=True, num_workers=2)

	source_train_loader = DataLoader(mnist_train, batch_size=batch_size)
	source_test_loader = DataLoader(mnist_test, batch_size=batch_size)


	cnn = CNN().to(device)

	optimizer = torch.optim.SGD(cnn.parameters(), lr=1.0, momentum=0.9)

	mu0, alpha, beta = 0.01, 10, 0.75
	scheduler = torch.optim.lr_scheduler.LambdaLR(
		optimizer,
		lambda e: 0.01 / (1 + alpha * e / epochs) ** beta
	)

	for epoch in range(epochs):
		train_loss = 0.

		for x, y in source_train_loader:
			x, y = x.cuda(), y.cuda()

			optimizer.zero_grad()
			logits = cnn(x)
			# print(logits.shape, y.shape)
			loss = F.cross_entropy(logits, y)
			loss.backward()
			optimizer.step()

			train_loss += loss.item()
		print(f'Epoch {epoch}, train loss: {round(train_loss / len(source_train_loader), 5)}')
		scheduler.step()
		print(f"\tLearning rate = {optimizer.param_groups[0]['lr']}")

		test_acc, test_loss = eval_model(cnn, source_test_loader)
		print(f"Test loss: {test_loss}, test acc: {test_acc}")

	return cnn, source_test_loader

# cnn, test_loader = train_mnist()