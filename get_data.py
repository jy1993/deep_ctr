from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import pickle


class Ali(Dataset):
	"""docstring for Ali"""
	def __init__(self, path, flag, feature_index_path):
		super(Ali, self).__init__()
		self.flag = flag
		self.cdata = open(f'{path}').read().strip().split('\n')
		# to remove the header
		self.cdata = self.cdata[1:]
		self.feature_index = pickle.load(open(feature_index_path, 'rb'))

	def __getitem__(self, index):
		terms = self.cdata[index].split(',')
		if self.flag == 'test':
			label = None
		else:
			label = int(terms[0])
		
		max_index = self.feature_index['features'][-1]
		data = np.array([0] * max_index)
		cat_features = terms[3:-4]
		all_f_index = []
		for cat_f in cat_features:
			kk_ = cat_f.split(' ')
			for k_ in kk_:
				all_f_index.append(int(k_.split(':')[1]))

		data[all_f_index] = 1
		return data, label

	def __len__(self):
		return len(self.cdata)

if __name__ == '__main__':
	ali = Ali('../data/val.csv', 'val', '../data/feature_index')	
