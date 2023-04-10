import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataclasses_json import dataclass_json
from dataclasses import dataclass


"""
Taken from this tutorial: https://nextjournal.com/gkoehler/pytorch-mnist
"""
@dataclass_json
@dataclass
class Hyperparameters(object):
    n_epochs: int = 5
    batch_size_train: int = 64
    batch_size_test: int = 1000
    learning_rate: float = 0.01
    momentum: float = 0.5
    log_interval: int = 6000

hp = Hyperparameters


random_seed = 6
torch.backends.cudnn.enabled = False



def load_data(batch_size_train, batch_size_test):
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(root='./data', train=True, download=True,
                                    transform=torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(
                                        (0.1307,), (0.3081,))
                                    ])),
        batch_size=batch_size_train, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(root='./data', train=False, download=True,
                                    transform=torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(
                                        (0.1307,), (0.3081,))
                                    ])),
        batch_size=batch_size_test, shuffle=True)
    return train_loader, test_loader


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


def train(epoch, network, optimizer, train_loader,train_losses, train_counter, log_interval):
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = network(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))

def run_wf(hp: Hyperparameters) -> torch.nn.Module:
    train_loader, test_loader = load_data(hp.batch_size_train, hp.batch_size_test)
    network = Net()
    optimizer = optim.SGD(network.parameters(), lr=hp.learning_rate, momentum=hp.momentum)
    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = [i*len(train_loader.dataset) for i in range(hp.n_epochs + 1)]
    for epoch in range(1, hp.n_epochs + 1):
        train(epoch, network, optimizer, train_loader,train_losses, train_counter, hp.log_interval)
    return network

if __name__=="__main__":
    run_wf(hp=hp)
