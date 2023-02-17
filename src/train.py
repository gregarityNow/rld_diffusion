
from .basis_funcs import *
from .diff_backbone import *;
from .viz import show_images
from .eval import do_evaluate


def save_images(schedule, epoch, class_emb_dim, w, model, timesteps):
	showLabels = torch.arange(10)
	image = sample(schedule, model, batch_size=10, labels=showLabels, w=w)

	suffix = getSuffix(class_emb_dim, w, epoch=epoch)
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

from torch.utils.data import DataLoader
def train_diff(cnn, test_loader, train_data, schedType = "sigmoid",model=None, class_emb_dim=None, w=0, epochs=30, timesteps = 200, quickie = 0):
	if schedType == "sigmoid":
		schedule = SigmoidSchedule(timesteps)
	elif schedType == "linear":
		schedule = LinearSchedule(timesteps);
	elif schedType == "quad":
		schedule = QuadraticSchedule(timesteps);
	else:
		raise Exception("Don't know scheduler",schedType);

	if model is None:
		model = Model(train_data.image_shape[0], train_data.channels, class_emb_dim=class_emb_dim)
	model.to(device)

	optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
	dataloader = DataLoader(train_data, batch_size=128, shuffle=True, pin_memory=True)
	criterion = torch.nn.MSELoss()

	for epoch in range(epochs):
		for step, (batch, labels) in enumerate(dataloader):

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

			loss.backward()
			optimizer.step()

			if step == 0:
				save_images(schedule, epoch, class_emb_dim, w, model, timesteps)

		if epoch % 10 == 0:
			do_evaluate(model, cnn, schedule, test_loader, w, quickie, epoch = epoch,
						schedType = schedType, class_emb_dim = class_emb_dim);

	do_evaluate(model, cnn, schedule, test_loader, w, quickie, epoch=epochs,
				schedType=schedType, class_emb_dim=class_emb_dim);
	save_images(schedule, epochs, class_emb_dim, w, model, timesteps)

	return model, schedule