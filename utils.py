def count_property(pred, item):
	'''
	aims to count how many common property labels pred and item have

	pred: a string   5799347067982556520:-1;509660095530134768:-1
	item: a string   2072967855524022579;5131280576272319091
	'''
	items = set(item.split(';'))
	pred = set([ele[1] for ele in pred.split(';') if ele != '-1'])
	common = items & pred
	return len(common)


if __name__ == '__main__':
	pred = "5799347067982556520:-1;509660095530134768:-1"
	item = "2072967855524022579;5131280576272319091"
	# split_time = 1537718400
	# train, val = df_train.loc[df_train['context_timestamp'] < 1537718400, :], df_train.loc[df_train['context_timestamp'] >= 1537718400, :]
	print(count_property(pred, item))