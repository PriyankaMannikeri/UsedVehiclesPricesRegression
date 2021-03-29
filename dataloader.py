import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np

class UsedVechiclesDataset(Dataset):

    def __init__(self, data, labels, norm=False):
        self.data = data
        self.labels = labels
        self.norm = norm
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # if self.transform is not None:
        #     curr_data = self.transform(self.data[idx, :])
        curr_data = torch.from_numpy(self.data[idx, :])
        curr_gt = torch.tensor(self.labels[idx])

        data_point = {"data": curr_data, "gt": curr_gt}
        return data_point
