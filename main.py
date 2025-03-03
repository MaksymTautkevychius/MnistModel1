import torch
from torch import nn
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from source.MnistModel1 import MnistModel1

epochs=10
BATCH_SIZE = 32
#optimize=
#loss_func=
#model1 = MnistModel1()

device = "cuda" if torch.cuda.is_available() else "cpu"

train_data = datasets.MNIST(
    root="data",
    train=True, 
    download=True, 
    transform=ToTensor(),
    target_transform=None 
)
class_names=train_data.classes

test_data = datasets.MNIST(
    root="data",
    train=False, 
    download=True,
    transform=ToTensor()
)
train_dataloader = DataLoader(train_data, 
    batch_size=BATCH_SIZE,
    shuffle=True 
)

test_dataloader = DataLoader(test_data,
    batch_size=BATCH_SIZE,
    shuffle=False 
)

loss_fn = nn.CrossEntropyLoss() 
optimizer = torch.optim.SGD(params=model1.parameters(), lr=0.1)

torch.manual_seed(42)
