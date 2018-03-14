import torch
from torch import nn
from torch.nn import functional as F

class NFM(nn.Module):
	"""docstring for NFM"""
	def __init__(self, name, vocab_size, embedding_size, hidden_zie=200, layers=1, p=0.5):
		super(NFM, self).__init__()
		self.name = name
		self.embedding = nn.Embedding(vocab_size, embedding_size)
		fc_layers = [nn.Linear(embedding_size, hidden_zie)]
		fc_layers.extend([nn.Linear(hidden_zie, hidden_zie)] * (layers - 1))
		self.last_layer = nn.Linear(hidden_zie, 1)
		self.fc_layers = nn.ModuleList(fc_layers)
		self.dropout = nn.Dropout(p=p, inplace=True)
		self.bi_batchnorm = nn.BatchNorm1d(embedding_size)
		self.batchnorm = nn.BatchNorm1d(hidden_zie)
		self.sigmoid = nn.Sigmoid()
		torch.manual_seed(1234)

	def forward(self, x):
		print('input: ', x.size())
		x = self.embedding(x)
		print('before: bi-interaction: ', x.size())
		x = [x[:, i, :] * x[:, j, :] for i in range(x.size(1)) for j in range(i + 1, x.size(1))]
		x_add = x[0]
		for x_ in x[1:]:
			x_add += x_
		#print('before cat:', len(x))
		#print(x[0])
		#x = torch.cat(x, 1)
		#print('after cat: ', x.size())
		#x = torch.sum(x, 1)
		print('after sum:', x_add.size())
		x = F.relu(self.bi_batchnorm(x_add))
		print(self.fc_layers)
		for fc_layer in self.fc_layers:
			x = self.dropout(F.relu(self.batchnorm(fc_layer(x))))
		x = self.last_layer(x)
		x = self.sigmoid(x)
		return x


