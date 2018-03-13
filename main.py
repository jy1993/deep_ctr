import torch
from models import deepfm
from ali_transform import get_data
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.autograd import Variable
import os
from torch import nn
import torchnet as tnt
from vis import Visualizer
from tqdm import tqdm


class Config():
	# data
	train_path = './data/train.csv'
	test_path = './data/test.csv'
	val_path = './data/val.csv'
	train_flag = True
	test_flag = False

	feature_index_path = './data/feature_index'

	# optimizier
	beta1 = 0.5
	beta2 = 0.999
	lr = 0.005
	weight_decay = 1e-5
	lr_decay = 0.95

	# training
	epochs = 10
	batch_size = 256

	# visualize and save
	print_every = 500
	eval_every = 1
	resume = False
	model_path = './save/models'
	output_path = './save/output'

def to_var(x, volatile=False):
	x = Variable(x)
	if torch.cuda.is_available():
		x = x.cuda()
	return x

def weight_init(m):
	if isinstance(m, nn.Linear):
		nn.init.xavier_uniform(m.weight.data)
		nn.init.xavier_uniform(m.bias.data)

config = Config()

def train(**kwargs):
	for k_, v_ in kwargs.items():
		setattr(config, k_, v_)

	vis_ = Visualizer()

	# data
	train_dataset = get_data.Ali(config.train_path, 'train', config.feature_index_path)
	val_dataset = get_data.Ali(config.val_path, 'val', config.feature_index_path)

	train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size, 
		shuffle=True, drop_last=True)
	val_loader = DataLoader(dataset=val_dataset, batch_size=config.batch_size)

	# model
	model = deepfm.FNN(config.feature_index_path)
	print(model)

	# print('initializing...')
	# model.apply(weight_init)

	# testing 
	if config.test_flag:
		test_dataset = get_data.Ali(config.test_path, 'test', config.feature_index_path)
		test_loader = DataLoader(dataset=test_dataset, batch_size=config.batch_size)
		model.load_state_dict(torch.load(os.path.join(config.model_path, '_best')))
		test(model, test_loader, config.output_path)

	# criterion and optimizer
	criterion = torch.nn.BCELoss()
	lr = config.lr
	optimizer = Adam(model.parameters(), lr=lr, betas=(config.beta1, config.beta2), 
		weight_decay = config.weight_decay)
	previous_loss = 1e6
	if torch.cuda.is_available():
		model.cuda()
		criterion.cuda()
	
	# meters
	loss_meter = tnt.meter.AverageValueMeter()
	# class_err = tnt.meter.ClassErrorMeter()
	# confusion_matrix = tnt.meter.ConfusionMeter(2, normalized=True)

	# val(model, val_loader, criterion)
	# resume training
	start = 0
	if config.resume:
		model_epoch = [int(fname.split('_')[-1]) for fname in os.listdir(config.model_path) 
			if 'best' not in fname]
		start = max(model_epoch)
		model.load_state_dict(torch.load(os.path.join(config.model_path, '_epoch_{start}')))
	if start >= config.epochs:
		print('Training already Done!')
		return 

	# train
	print('start training...')
	for i in range(start, config.epochs):
		loss_meter.reset()
		# class_err.reset()
		# confusion_matrix.reset()
		for ii, (c_data, labels) in tqdm(enumerate(train_loader)):
			c_data = to_var(c_data)
			labels = to_var(labels).float()
			# labels = labels.view(-1, 1)

			pred = model(c_data)
			# print(pred, labels)
			loss = criterion(pred, labels)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			# meters update and visualize
			loss_meter.add(loss.data[0])
			# confusion_matrix.add(pred.data.squeeze(), labels.data.type(torch.LongTensor))

			if (ii + 1) % config.print_every == 0:
				vis_.plot('train_loss', loss_meter.value()[0])
				print(f'''epochs: {i + 1}/{config.epochs} batch: {ii + 1}/{len(train_loader)}
							train_loss: {loss.data[0]}''')

		print('evaluating...')
		# train_cm = confusion_matrix.value()
		val_cm, val_accuracy, val_loss = val(model, val_loader, criterion)
		vis_.plot('val_loss', val_loss)
		vis_.log(f"epoch:{start + 1},lr:{lr},loss:{val_loss}")

		torch.save(model.state_dict(), os.path.join(config.model_path, f'_epoch_{i}'))

		# update learning rate
		if loss_meter.value()[0] > previous_loss:  
			torch.save(model.state_dict(), os.path.join(config.model_path, '_best'))
			lr = lr * config.lr_decay
			for param_group in optimizer.param_groups:
				param_group['lr'] = lr
		previous_loss = loss_meter.value()[0]

def val(fmodel, data_loader, loss_f):
	fmodel.eval()
	confusion_matrix = tnt.meter.ConfusionMeter(2)
	loss_meter = tnt.meter.AverageValueMeter()
	loss_meter.reset()
	for ii, (c_data, labels) in tqdm(enumerate(data_loader)):
		c_data, labels = to_var(c_data), to_var(labels).float()
		pred = fmodel(c_data)
		loss = loss_f(pred, labels)
		confusion_matrix.add(pred.data.squeeze(), labels.data.type(torch.LongTensor))
		loss_meter.add(loss.data[0])
	fmodel.train()
	cm_value = confusion_matrix.value()
	accuracy = 100. * (cm_value[0][0] + cm_value[1][1]) / (cm_value.sum())
	return confusion_matrix, accuracy, loss_meter.value()[0] 

def test(fmodel, data_loader, output_path):
	fmodel.eval()
	preds = []
	for (c_data, _) in data_loader:
		c_data = to_var(c_data, volatile=True)
		preds.append(fmodel(c_data))
	print(preds)

if __name__ == '__main__':
	# from fire import Fire
	# Fire()
	train()