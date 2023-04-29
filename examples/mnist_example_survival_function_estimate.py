import os
from functools import cached_property

import numpy as np
import torch
from torch import nn
from torchvision.datasets.mnist import MNIST

from survivalreg.dataset import SurvivalFuncEstimateDataset, Sample
from survivalreg.trainer import Trainer


class DummyData():
    def __init__(self, testing=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        mnist = MNIST('.output', download=True, train=not testing)
        rnd = np.random.RandomState(234523)

        label = mnist.targets
        data = mnist.data
        self.label = label
        self.data = data
        rnd1 = rnd.randint(1, 10, len(self.label))
        self.follow_up_1 = np.array(
            [min(rnd1[i], max(1, 11-self.label[i]-rnd1[i])) for i in range(len(rnd1))])
        self.follow_up_2 = rnd.randint(1, 10, len(self.label))
        self.follow_up_2 = self.follow_up_2 + self.follow_up_1

    def __getitem__(self, item: int):
        return (
            torch.unsqueeze(self.data[item], 0) / 255,
            self.follow_up_1[item],
            self.follow_up_2[item],
            self.follow_up_2[item]+self.label[item] > 11)

    def __len__(self):
        return len(self.label)


class DeepSurModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        rnd = np.random.RandomState(53236)
        b = torch.FloatTensor(abs(rnd.normal(0, 10, (1, 512))+0.1))
        k = torch.FloatTensor(abs(rnd.normal(0, 10, (1, 512))+1))
        self.register_buffer('b', b)
        self.register_buffer('k', k)

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.AvgPool2d(2, 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.AvgPool2d(2, 2),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc3 = nn.Linear(128, 512)

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        x = self.fc3(x)
        x = nn.functional.softmax(x, dim=1)
        return x


class TrainMNIST(Trainer):

    @cached_property
    def model(self):
        return DeepSurModel().to(self.device)

    @cached_property
    def beta(self):
        return 3.0

    @property
    def train_dataset(self):
        return DummyData()

    @property
    def test_dataset(self):
        return DummyData(testing=True)

    def batch(self, epoch, i_batch, data) -> dict:
        imgs, t1, t2, e = data
        imgs = imgs.to(self.device)
        t1 = t1.to(self.device)
        t2 = t2.to(self.device)
        e = e.to(self.device)
        t1 = torch.unsqueeze(t1, 1)
        t2 = torch.unsqueeze(t2, 1)
        feat = self.model(imgs)
        F1 = 1 - torch.exp(-(1/self.model.b * t1) ** self.model.k)
        F2 = 1 - torch.exp(-(1/self.model.b * t2) ** self.model.k)
        P1 = (feat * F1).sum(dim=1) * 0.99 + 0.005
        P2 = (feat * F2).sum(dim=1) * 0.99 + 0.005
        loss = torch.log(P1) + torch.log(1-P2) * self.beta * e
        x = torch.linspace(.1, 10, 6)
        x = x.unsqueeze(dim=1)
        x = x.to(self.device)
        ppred = 1 - torch.exp(-(1/self.model.b * x) ** self.model.k)
        ppred = ppred.unsqueeze(dim=0)
        feat = feat.unsqueeze(dim=1)
        ppred = ppred * feat
        ppred = ppred.sum(dim=2)
        return dict(
            loss=loss.mean(),
            pred=ppred
        )

    def matrix(self, epoch, data) -> dict:
        print(data['pred'])
        return dict(
            loss=float(data['loss'].mean())
        )


if __name__ == '__main__':
    os.environ['batch_size'] = '128'
    os.environ['lr'] = '0.00001'
    trainer = TrainMNIST()
    trainer.train()