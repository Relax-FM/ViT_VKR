import torch
from torch.utils.data import Dataset

class DataSet(Dataset):
    def __init__(self, data=None):
        if (data==None):
            raise Exception("Plz, entry all data")
        else:
            self.data = data

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, index):
        # label так и оставляем, а вот info надо исправить
        info = torch.tensor(self.data[0][index])  # [i, [4, 50]]
        label = torch.tensor(self.data[1][index])  # [i, 1]
        info = info.to(torch.float32)
        label = label.to(torch.float32)
        return {
            'info': info,
            'label': label
        }
