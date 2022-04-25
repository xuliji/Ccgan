import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_dim, 128), nn.LeakyReLU(0.2),
            nn.Linear(128, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 512), nn.LeakyReLU(0.2),
            nn.Linear(512, out_dim), nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

class Generator(nn.Module):
    def __init__(self, label_dim=5, label_emb_dim=32, noise_dim=100):
        super(Generator, self).__init__()

        self.label_emb = MLP(label_dim, label_emb_dim)

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(noise_dim + label_emb_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, 2*201),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        # Concatenate label embedding and image to produce input
        gen_input = torch.cat((self.label_emb(labels), noise), -1)
        shapes = self.model(gen_input)
        shapes = shapes.view(shapes.size(0), -1, 2)
        return shapes


class Discriminator(nn.Module):
    def __init__(self, label_dim=5, label_emb_dim=32):
        super(Discriminator, self).__init__()

        self.label_embedding = MLP(label_dim, label_emb_dim)

        self.model = nn.Sequential(
            nn.Linear(label_emb_dim + 2*201, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
        )

    def forward(self, shapes, labels):
        # Concatenate label embedding and image to produce input
        d_in = torch.cat((shapes.view(shapes.size(0), -1), self.label_embedding(labels)), -1)
        validity = self.model(d_in)
        return validity

