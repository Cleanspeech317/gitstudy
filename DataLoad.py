import random
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

# class MyDataset(Dataset):
# 	def __init__(self, path, num_user, num_item):
# 		super(MyDataset, self).__init__()
# 		self.data = np.load(path+'train.npy')
# 		self.adj_lists = np.load(path+'adj.npy').item()
# 		self.all_set = set(range(num_user, num_user+num_item))
#
# 	def __getitem__(self, index):
# 		user, pos_item = self.data[index]
# 		neg_item = random.sample(self.all_set.difference(self.adj_lists[user]), 1)[0]
# 		return [user, pos_item, neg_item]
#
# 	def __len__(self):
# 		return len(self.data)

class MyDataset(Dataset):
    def __init__(self, num_user, num_item, user_item_dict, edge_index):
        self.edge_index = edge_index
        self.num_user = num_user
        self.num_item = num_item
        self.user_item_dict = user_item_dict
        self.all_set = set(range(num_user, num_user+num_item))

    def __len__(self):
        return len(self.edge_index)

    def __getitem__(self, index):
        user, pos_item = self.edge_index[index]
        while True:
            neg_item = random.sample(self.all_set, 1)[0]
            if neg_item not in self.user_item_dict[user]:
                break
        return torch.LongTensor([user,user]), torch.LongTensor([pos_item, neg_item])

# if __name__ == '__main__':
# 	num_item = 5986
# 	num_user = 55485
# 	dataset = MyDataset('./Data/', num_user, num_item)
# 	dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
#
# 	for data in dataloader:
# 		user, pos_items, neg_items = data
# 		#解释了data怎么来的
# 		print(user)


