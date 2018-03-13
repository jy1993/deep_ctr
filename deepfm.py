from torch import nn
import numpy as np
import torch
import torch.nn.functional as F
import pickle

class FNN(nn.Module):
	"""docstring for FNN"""
	def __init__(self, feature_index_path, k=30, hidden1=200, hidden2=200):
		super(FNN, self).__init__()
		feature_index = pickle.load(open(feature_index_path, 'rb'))
		self.features = feature_index['features']
		self.num_fields = feature_index['fields'][-1]
		# print(self.features, self.num_fields)
		num_f_in_field = [self.features[i + 1] - self.features[i] for i in range(self.num_fields)]
		# print(num_f_in_field)
		# print(self.f_num_2_dim)
		# embedding_dims = [self.f_num_2_dim(ele, max_dim) for ele in num_f_in_field]
		# print(embedding_dims)
		fm = [nn.Linear(num_f_in_field[i], k) for i in range(self.num_fields)]
		self.embedding = nn.ModuleList(fm)
		self.fc1 = nn.Linear(k * self.num_fields, hidden1)
		self.fc2 = nn.Linear(hidden1, hidden2)
		self.fc3 = nn.Linear(hidden2, 1)
		self.batchnorm1 = nn.BatchNorm1d(hidden1)
		self.batchnorm2 = nn.BatchNorm1d(hidden2)
		self.dropout = nn.Dropout(inplace=True)

	def forward(self, x):
		cut_x = [x[:, self.features[ii]:self.features[ii + 1]] for ii in range(self.num_fields)]
		cut_x = [ele.float() for ele in cut_x]
		out = [part(cut_x[ii]) for ii, part in enumerate(self.embedding)]
		out = torch.cat(out, 1)
		out = self.dropout(F.relu(self.batchnorm1(self.fc1(out))))
		out = self.dropout(F.relu(self.batchnorm1(self.fc2(out))))
		out = F.sigmoid(self.fc3(out))
		return out.view(-1)

	def f_num_2_dim(self, num, max_dim):
		if num >= max_dim:
			return max_dim
		else:
			return int(np.floor((num + 1) / 2 ))

if __name__ == '__main__':
	model = FNN('../data/feature_index')



		

		