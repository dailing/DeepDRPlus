from model import ModelProgression
from torch import nn
import torch
import numpy as np
from functools import cached_property
from trainer import Trainer
from torch.utils.data import Dataset
import pandas as pd
import cv2
import albumentations as aug
import albumentations.pytorch as aug_torch


class DeepSurModel(nn.Module):
    def __init__(self, K=512) -> None:
        super().__init__()
        self.K = K
        # sample parameters for the mixture model
        rnd = np.random.RandomState(12345)
        b = torch.FloatTensor(abs(rnd.normal(0, 10, (1, 1, self.K))+5.0))
        k = torch.FloatTensor(abs(rnd.normal(0, 10, (1, 1, self.K))+5.0))
        self.register_buffer('b', b)
        self.register_buffer('k', k)

        self.cnn = ModelProgression()

    def _pdf_at(self, t):
        # pdf: nBatch * n * K
        pdf = 1 - torch.exp(-(1/self.b * (t)) ** self.k)
        return pdf

    def calculate_pdf(self, w, t):
        """
        Calculates the probability distribution function (pdf) for the given 
        data.
        
        param w: nBatch * K: weights for mixture model
        param t: nBatch * n: target time to calculate pdf at
        return: nBatch * n: pdf values
        """
        t = t.unsqueeze(dim=2)
        w = nn.functional.softmax(w, dim=1)
        w = w.unsqueeze(dim=1)
        pdf = self._pdf_at(t)
        pdf = pdf * w
        pdf = pdf.sum(dim=2)
        return pdf

    def forward(self, x, t=None):
        x = self.cnn(x)
        if t is None:
            return x
        return x, self.calculate_pdf(x, t)


class ProgressionData(Dataset):

    def __init__(self, datasheet, transform):
        super().__init__()
        self.df = pd.read_csv(datasheet)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_file = self.df.iloc[idx]['image']
        image = cv2.imread(img_file, cv2.IMREAD_COLOR)
        image = self.transform(image=image)['image']
        return (
            image,
            self.df.iloc[idx]['t1'],
            self.df.iloc[idx]['t2'],
            self.df.iloc[idx]['e']
        )


class TrainerDR(Trainer):

    @cached_property
    def model(self):
        return DeepSurModel().to(self.device)

    @cached_property
    def beta(self):
        return 1

    @cached_property
    def train_dataset(self):
        transform = aug.Compose([
            aug.SmallestMaxSize(max_size=self.cfg.image_size, always_apply=True),
            aug.CenterCrop(self.cfg.image_size, self.cfg.image_size,
                           always_apply=True),
            aug.Flip(p=0.5),
            aug.ImageCompression(quality_lower=10, quality_upper=80, p=0.2),
            aug.MedianBlur(p=0.3),
            aug.RandomBrightnessContrast(p=0.5),
            aug.RandomGamma(p=0.2),
            aug.GaussNoise(p=0.2),
            aug.Rotate(border_mode=cv2.BORDER_CONSTANT,
                       value=0, p=0.7, limit=45),
            aug.ToFloat(always_apply=True),
            aug_torch.ToTensorV2(),
        ])
        return ProgressionData('data/train.csv', transform)

    @cached_property
    def test_dataset(self):
        transform = aug.Compose([
            aug.SmallestMaxSize(max_size=self.cfg.image_size, always_apply=True),
            aug.CenterCrop(self.cfg.image_size, self.cfg.image_size,
                           always_apply=True),
            aug.ToFloat(always_apply=True),
            aug_torch.ToTensorV2(),
        ])
        return ProgressionData('data/test.csv', transform)

    @cached_property
    def optimizer(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.cfg.lr, weight_decay=1e-5)
        return optimizer

    def batch(self, epoch, i_batch, data) -> dict:
        # get and prepare data elements
        imgs, t1, t2, e = data
        imgs = imgs.to(self.device)
        t1 = t1.to(self.device)
        t2 = t2.to(self.device)
        e = e.to(self.device)

        w, P = self.model(imgs, torch.stack([t1, t2], dim=1))
        P1 = P[:, 0]
        P2 = P[:, 1]
        loss = (P1 + 0.000001) + (1-P2 + 0.000001) * self.beta * (t2 > t1)
        loss += torch.abs(w).mean() * 0.001

        return dict(
            loss=loss.mean(),
        )

    def matrix(self, epoch, data) -> dict:
        return dict(
            loss=float(data['loss'].mean())
        )


if __name__ == '__main__':
    trainer = TrainerDR()
    trainer.train()
