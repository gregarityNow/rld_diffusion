from .basis_funcs import *;

from torch.utils.data import DataLoader


class CNN(nn.Module):
	def __init__(self,channels = 1):
		super().__init__()
		self.net = nn.Sequential(
			nn.Conv2d(channels, 64 ,(5 ,5), stride=1, padding=2),
			nn.ReLU(),
			nn.MaxPool2d((2 ,2)),
			nn.Conv2d(64, 32, (5 ,5), stride=1, padding=2),
			nn.ReLU(),
			nn.MaxPool2d((2 ,2)),
			nn.Conv2d(32, 16, (5 ,5), stride=1, padding=2),
			nn.ReLU(),
			nn.MaxPool2d((2 ,2)),
		)
		if channels == 3:
			self.classif = nn.Sequential(
				nn.Linear(256, 50), nn.ReLU(),nn.Linear(50,10)
			)
		else:
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
def train_cnn(epochs = 5, quickie = False, dsName = "MNIST"):

	if dsName == "MNIST":
		pixels = torchvision.datasets.MNIST(train=True, download=True,root=dataLoc).data / 255
		mean = pixels.mean().item()
		std = pixels.std().item()
		channels = 1
		mean = torch.tensor([mean])
		std = torch.tensor([std])
	elif dsName == "CIFAR10":
		pixels = torchvision.datasets.CIFAR10(train=True, download=True, root=dataLoc).data / 255
		mean = pixels.mean(axis=(0, 1, 2))
		std = pixels.std(axis=(0, 1, 2))
		channels = 3
		mean = torch.tensor(mean)
		std = torch.tensor(std)
	else:
		raise Exception("Don't know ds",dsName)


	print(f"Mean {mean} and Std {std}")


	transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize(mean, std)
	])


	if dsName == "MNIST":
		train_data = torchvision.datasets.MNIST(train=True, transform=transform, download=True, root=dataLoc)
		test_data = torchvision.datasets.MNIST(train=False, transform=transform, download=True, root=dataLoc)
	elif dsName == "CIFAR10":
		train_data = torchvision.datasets.CIFAR10(train=True, transform=transform, download=True, root=dataLoc)
		test_data = torchvision.datasets.CIFAR10(train=False, transform=transform, download=True, root=dataLoc)
	else:
		raise Exception("Don't know ds",dsName)

	if quickie:
		train_data = torch.utils.data.Subset(train_data,torch.arange(100))
		test_data = torch.utils.data.Subset(test_data, torch.arange(100))

	source_train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
	source_test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

	cnn = CNN(channels).to(device)

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

	return cnn, source_train_loader, source_test_loader

# cnn, test_loader = train_mnist()