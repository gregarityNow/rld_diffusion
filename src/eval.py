

from .basis_funcs import *;


@torch.no_grad()
def calculate_inception_score(logits, eps=1e-20):
	# inspired by https://machinelearningmastery.com/how-to-implement-the-inception-score-from-scratch-for-evaluating-generated-images/
	p_yx = torch.softmax(logits, 1)
	p_y = torch.ones(10) / 10
	kld = p_yx * (torch.log(p_yx + eps) - torch.log(p_y + eps))
	mean_kld = kld.sum() / logits.shape[0]
	IS_score = torch.exp(mean_kld)
	return IS_score.item()


from scipy.linalg import sqrtm


@torch.no_grad()
def calculate_FID(real_activations, fake_activations):
	# inspired by https://machinelearningmastery.com/how-to-implement-the-frechet-inception-distance-fid-from-scratch/#:~:text=A%20lower%20FID%20indicates%20better,of%20random%20noise%20and%20blur.
	muReal, covReal = real_activations.mean(axis=0), torch.cov(real_activations.T)
	muFake, covFake = fake_activations.mean(axis=0), torch.cov(fake_activations.T)

	meanDiffSq = torch.sum((muReal - muFake) ** 2)
	# calculate sqrt of product between cov

	sqrtPart = 2 * sqrtm(covReal @ covFake).real
	# print("dell boca vista",covReal.shape,covReal[torch.isnan(covReal)],covFake[torch.isnan(covFake)],sqrtPart[torch.isnan(sqrtPart)].shape)
	traceInnards = covReal + covFake - sqrtPart
	# calculate score
	fid = meanDiffSq + torch.trace(traceInnards)
	return fid.item()


@torch.no_grad()
def do_evaluate(diffModel, cnn,schedule, test_loader, w, quickie, epoch, class_emb_dim, schedType,version, loss, dsName):
	fid_score, is_score = evaluate_diff_model(diffModel, cnn, test_loader, w, schedule,
											  numFakeIters=(50 if not quickie else 2), batch_size=100)

	d = {"w": w, "class_emb_dim": class_emb_dim, "schedType": schedType, "fid": fid_score,
		 "is_score": is_score,"epoch":epoch,"version":version,"loss":loss, "dsName":dsName}
	dumpRes(d);

import torch.utils.data as data_utils

from .diff_backbone import sample
@torch.no_grad()
def evaluate_diff_model(model, cnn, test_loader, w, schedule, numFakeIters=10,batch_size=100):
	fakes = []

	showLabels = torch.arange(10).repeat(int(batch_size / 10))

	for _ in tqdm(range(numFakeIters)):
		fakeSample = sample(schedule, model, batch_size=batch_size, labels=showLabels, w=w, justLast=True)
		print("fakeSamp",len(fakeSample),fakeSample[0].shape);
		fakes.extend(fakeSample)

	fakes = torch.cat(fakes)#.squeeze(0)
	print("shakes", fakes.shape)
	fakes = data_utils.TensorDataset(fakes)
	fakes_loader = data_utils.DataLoader(fakes, batch_size=batch_size, shuffle=True)
	fakeAct, fakeLog = [], []
	for batch in fakes_loader:
		batch = batch[0]
		act, logits = cnn.get_act_and_class(batch.to(device))
		fakeAct.append(act)
		fakeLog.append(logits)
	fakeAct, fakeLog = torch.cat(fakeAct), torch.cat(fakeLog)

	testAct, testLog = [], []
	for batch, _ in test_loader:
		act, logits = cnn.get_act_and_class(batch.to(device))
		testAct.append(act)
		testLog.append(logits)
	testAct, testLog = torch.cat(testAct), torch.cat(testLog)
	print("fakt", testAct.shape, fakeAct.shape, fakeLog.shape)

	fid_score = calculate_FID(testAct, fakeAct)
	is_score = calculate_inception_score(fakeLog)
	fid_score_same = calculate_FID(fakeAct, fakeAct)
	is_score_test = calculate_inception_score(testLog)

	print("DIFF: fid score", fid_score, "same:", fid_score_same, "is score", is_score, "is score test", is_score_test)

	return fid_score, is_score


