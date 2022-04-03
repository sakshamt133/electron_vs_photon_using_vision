from torch.utils.data import random_split, DataLoader
from dataset import Data
import utils

data = Data(
    utils.electron_path, utils.proton_path, utils.transform
)

train_size = int(len(data) * 0.9)
test_size = len(data) - train_size

train_data, test_data = random_split(data, [train_size, test_size])

train_batch = DataLoader(
    train_data,
    batch_size=32,
    shuffle=True,
    num_workers=8
)

test_batch = DataLoader(
    test_data,
    batch_size=32,
    shuffle=True,
    num_workers=8
)

for (img, labels) in train_batch:
    print(img.shape, labels.shape)