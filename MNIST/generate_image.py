import argparse
import os
import numpy as np
import math
import itertools

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
from torchvision.utils import make_grid
from torch.nn.utils import spectral_norm

import torch.nn as nn
import torch.nn.functional as F
import torch


os.makedirs("test/varying_c1/", exist_ok=True)
os.makedirs("test/varying_c2/", exist_ok=True)
os.makedirs("test/varying_c3/", exist_ok=True)
os.makedirs("test/varying_c4/", exist_ok=True)
os.makedirs("test/varying_c5/", exist_ok=True)
os.makedirs("test/varying_c6/", exist_ok=True)
os.makedirs("test/varying_c7/", exist_ok=True)


parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=128, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=62, help="dimensionality of the latent space")
parser.add_argument("--code_dim", type=int, default=7, help="latent code")
parser.add_argument("--n_classes", type=int, default=10, help="number of classes for dataset")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=4000, help="interval between image sampling")
opt = parser.parse_args()

cuda = True if torch.cuda.is_available() else False

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        input_dim = opt.latent_dim + opt.n_classes + opt.code_dim

        self.init_size = opt.img_size // 4  # Initial size before upsampling
        self.l1 = nn.Sequential(nn.Linear(input_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, opt.channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise, labels, code):
        gen_input = torch.cat((noise, labels, code), -1)
        out = self.l1(gen_input)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


def to_categorical(y, num_columns):
    """Returns one-hot encoded Variable"""
    y_cat = np.zeros((y.shape[0], num_columns))
    y_cat[range(y.shape[0]), y] = 1.0

    return Variable(FloatTensor(y_cat))


static_label = []
for i in range(10):
    for j in range(10):
        static_label.append(i)

static_label = to_categorical(np.asarray(static_label), num_columns=opt.n_classes)

static_z = Variable(FloatTensor(np.zeros((opt.n_classes* 10, opt.latent_dim))))


def sample_image(n_row, batches_done):
    """Saves a grid of generated digits ranging from 0 to n_classes"""


    # Get varied c1 and c2
    zeros = np.zeros((n_row * n_row, 1))
    #c_varied = np.repeat(np.linspace(-2, 2, n_row)[:, np.newaxis], n_row, 0)
    c_varied = -np.tile(np.linspace(-1, 1, n_row), n_row)[:, np.newaxis]
    c1 = Variable(FloatTensor(np.concatenate((c_varied, zeros, zeros, zeros, zeros, zeros, zeros), -1)))
    c2 = Variable(FloatTensor(np.concatenate((zeros, c_varied, c_varied, zeros, zeros, zeros, zeros), -1)))
    c3 = Variable(FloatTensor(np.concatenate((zeros, zeros, c_varied, zeros, zeros, zeros, zeros), -1)))
    c4 = Variable(FloatTensor(np.concatenate((zeros, zeros, zeros, c_varied, zeros, zeros, zeros), -1)))
    c5 = Variable(FloatTensor(np.concatenate((zeros, zeros, zeros, zeros, c_varied, zeros, zeros), -1)))
    c6 = Variable(FloatTensor(np.concatenate((zeros, zeros, zeros, zeros, zeros, c_varied, zeros), -1)))
    c7 = Variable(FloatTensor(np.concatenate((zeros, zeros, zeros, zeros, zeros, zeros, c_varied), -1)))



    
    sample1 = generator(static_z, static_label, c1)
    sample2 = generator(static_z, static_label, c2)
    sample3 = generator(static_z, static_label, c3)
    sample4 = generator(static_z, static_label, c4)
    sample5 = generator(static_z, static_label, c5)
    sample6 = generator(static_z, static_label, c6)
    sample7 = generator(static_z, static_label, c7)

    grid_1 = make_grid(sample1.data, nrow =n_row)
    grid_2 = make_grid(sample2.data, nrow =n_row)
    grid_3 = make_grid(sample3.data, nrow =n_row)
    grid_4 = make_grid(sample4.data, nrow =n_row)
    grid_5 = make_grid(sample5.data, nrow =n_row)
    grid_6 = make_grid(sample6.data, nrow =n_row)
    grid_7 = make_grid(sample7.data, nrow =n_row)



    save_image(grid_1, "test/varying_c1/%d.png" % batches_done, nrow=n_row, normalize=True)
    save_image(grid_2, "test/varying_c2/%d.png" % batches_done, nrow=n_row, normalize=True)
    save_image(grid_3, "test/varying_c3/%d.png" % batches_done, nrow=n_row, normalize=True)
    save_image(grid_4, "test/varying_c4/%d.png" % batches_done, nrow=n_row, normalize=True)
    save_image(grid_5, "test/varying_c5/%d.png" % batches_done, nrow=n_row, normalize=True)
    save_image(grid_6, "test/varying_c6/%d.png" % batches_done, nrow=n_row, normalize=True)
    save_image(grid_7, "test/varying_c7/%d.png" % batches_done, nrow=n_row, normalize=True)




generator = Generator()
PATH = "generator_40000.pt"
#generator = nn.DataParallel(generator)
generator.cuda()
generator.load_state_dict(torch.load(PATH))
generator.eval()
generator.requires_grad = False

sample_image(10, 0)
