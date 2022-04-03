import h5py
import torch
import numpy as np
from torch.utils.data import Dataset


class Data(Dataset):
    def __init__(self, e_path, p_path, transform=None):
        super(Data, self).__init__()
        with h5py.File(p_path, mode='r') as hdf:
            self.photons = np.array(hdf.get('X'))

        with h5py.File(e_path, mode='r') as hdf:
            self.electrons = np.array(hdf.get('X'))

        data = []
        for val in self.electrons:
            val = transform(val)
            data.append((val, 1))

        for val in self.photons:
            val = transform(val)
            data.append((val, 0))

        data = np.array(data, dtype=object)
        print(data.shape, data[0].shape, data[1].shape)
        self.data = torch.from_numpy(data)

    def __getitem__(self, item):
        return torch.Tensor(self.data[item][0]), torch.Tensor(self.data[item][1])

    def __len__(self):
        return self.photons.shape[0] + self.electrons.shape[0]
