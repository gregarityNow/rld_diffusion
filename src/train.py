from numpy import mean
from .basis_funcs import *
from .diff_backbone import *;
from .viz import show_images
from .eval import do_evaluate


def save_images(schedule, epoch, class_emb_dim, w, model, timesteps,version, dsName):
	showLabels = torch.arange(10)
	image = sample(schedule, model, batch_size=10, labels=showLabels, w=w)

	suffix = getSuffix(class_emb_dim, w, epoch=epoch,version=version, dsName = dsName)
	show_images(
		torch.cat(
			[
				image[-timesteps // 2],
				image[-timesteps // 3],
				image[-timesteps // 4],
				image[-1],
			]
		), labels=showLabels, nrow=10, suffix=suffix
	)

def train_diff(cnn, test_loader, train_loader, schedType = "sigmoid",model=None,version = 0,
			   dsName = "MNIST",class_emb_dim=None, w=0, epochs=30, timesteps = 200, quickie = 0):
	if schedType == "sigmoid":
		schedule = SigmoidSchedule(timesteps)
	elif schedType == "linear":
		schedule = LinearSchedule(timesteps);
	elif schedType == "quad":
		schedule = QuadraticSchedule(timesteps);
	else:
		raise Exception("Don't know scheduler",schedType);

	if dsName == "MNIST":
		channels = 1
		imageWidth = 28
	elif dsName == "CIFAR10":
		channels = 3
		imageWidth = 32
	else:
		raise  Exception("Don't know ds",dsName);

	if model is None:
		model = Model(imageWidth, channels, class_emb_dim=class_emb_dim)
	model.to(device)

	optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
	criterion = torch.nn.MSELoss()

	for epoch in range(epochs):
		epochLoss = []
		model.train()
		for step, (batch, labels) in enumerate(train_loader):

			optimizer.zero_grad()

			batch_size = batch.shape[0]
			batch = batch.to(device)
			if class_emb_dim is None:
				labelsMod = None
			else:
				labelsMod = labels.to(device)

			ts = torch.randint(low=0, high=schedule.timesteps, size=(batch_size,), device=device)
			batch_with_noise_added, noise = q_sample(schedule, batch, t=ts)

			targetGaussian = model(batch_with_noise_added, ts, labelsMod)
			loss = criterion(targetGaussian, noise)

			if step % 100 == 0:
				print("epoch",epoch,"step",step,"Loss:", loss.item())
			if loss.item() == loss.item():
				loss.backward()
				optimizer.step()
				epochLoss.append(loss.item())
			else:
				print("skipping this update, we have a nan loss..")

		if epoch % 10 == 0:
			do_evaluate(model, cnn, schedule, test_loader, w, quickie, epoch = epoch,loss = mean(epochLoss),
						version=version,schedType = schedType, class_emb_dim = class_emb_dim, dsName = dsName);
			save_images(schedule, epoch, class_emb_dim, w, model, timesteps, version=version, dsName = dsName)

	do_evaluate(model, cnn, schedule, test_loader, w, quickie, epoch=epochs,version=version,
				loss = mean(epochLoss),schedType=schedType, class_emb_dim=class_emb_dim, dsName = dsName);
	save_images(schedule, epochs, class_emb_dim, w, model, timesteps,version=version, dsName = dsName)

	return model, schedule