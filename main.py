from vision_tranformer import VisionTransformer
import torch
from train_test_batch import train_batch

loss = torch.nn.CrossEntropyLoss()
model = VisionTransformer()
opti = torch.optim.Adam(model.parameters(), lr=0.001)
epochs = 1

if __name__ == '__main__':
    for epoch in range(epochs):
        for (img, labels) in train_batch:
            y_hat = model(img)
            l = loss(y_hat, labels)
            print(f"for epoch {epoch} loss is {l}")
            l.backward()
            opti.step()
            opti.zero_grad()
