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


os.makedirs("images/static/", exist_ok=True)
os.makedirs("images/original/", exist_ok=True)
os.makedirs("images/scaled/", exist_ok=True)
os.makedirs("images/varying_c1/", exist_ok=True)
os.makedirs("images/varying_c2/", exist_ok=True)
os.makedirs("images/varying_c3/", exist_ok=True)
os.makedirs("images/varying_c4/", exist_ok=True)
os.makedirs("images/varying_c5/", exist_ok=True)
os.makedirs("images/varying_c6/", exist_ok=True)
os.makedirs("images/varying_c7/", exist_ok=True)
os.makedirs("images/varying_c8/", exist_ok=True)


parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=16, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=200, help="dimensionality of the latent space")
parser.add_argument("--code_dim", type=int, default=8, help="latent code")
parser.add_argument("--n_classes", type=int, default=10, help="number of classes for dataset")
parser.add_argument("--img_size", type=int, default=64, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=4000, help="interval between image sampling")
opt = parser.parse_args()
print(opt)

cuda = True if torch.cuda.is_available() else False



def to_categorical(y, num_columns):
    """Returns one-hot encoded Variable"""
    y_cat = np.zeros((y.shape[0], num_columns))
    y_cat[range(y.shape[0]), y] = 1.0

    return Variable(FloatTensor(y_cat))




class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        input_dim = opt.latent_dim + opt.code_dim + opt.n_classes

        self.init_size = opt.img_size // 2**4  # Initial size before upsampling
        #self.l1 = nn.Sequential(nn.Linear(input_dim, 128 * self.init_size ** 2))
        

        self.conv_blocks = nn.Sequential(
            nn.ConvTranspose2d(input_dim, 1024, 4, 1, 0),

            nn.ConvTranspose2d(1024, 512, 4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(128, opt.channels, 4, stride=2, padding=1),
            nn.Tanh(),
        )


    def forward(self, noise, labels, code):
        #gen_input = noise
        gen_input = torch.cat((noise, labels, code), -1)

        gen_input = gen_input.view(gen_input.size(0), gen_input.size(1), 1, 1)
        img = self.conv_blocks(gen_input)

        return img



PATH = "checkpoint_600000.tar"
print ("loading checkpoint")
print (PATH)
    
checkpoint = torch.load(PATH)

generator = Generator()
generator.load_state_dict(checkpoint['generator_state_dict'])
generator.eval()


if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")

if cuda:
    generator.cuda()


# Configure data loader


FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

# Static generator inputs for sampling
static_z = Variable(FloatTensor(np.zeros((opt.n_classes ** 2, opt.latent_dim))))
static_label = to_categorical(
    np.array([num for _ in range(opt.n_classes) for num in range(opt.n_classes)]), num_columns=opt.n_classes
)
static_code = Variable(FloatTensor(np.zeros((opt.n_classes ** 2, opt.code_dim))))

def transpose_row_col(sample):
    sample_trans = sample.clone()
    sample_trans[0] = sample[0]
    sample_trans[1] = sample[3]
    sample_trans[2] = sample[6]
    sample_trans[3] = sample[1]
    sample_trans[4] = sample[4]
    sample_trans[5] = sample[7]
    sample_trans[6] = sample[2]
    sample_trans[7] = sample[5]
    sample_trans[8] = sample[8]

    return sample_trans


static_z = Variable(FloatTensor(np.zeros((opt.n_classes ** 2, opt.latent_dim))))
static_label = to_categorical(
    np.array([num for _ in range(opt.n_classes) for num in range(opt.n_classes)]), num_columns=opt.n_classes
)

def sample_image(n_row = 10, batches_done = 0):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    

    # Get varied c1 and c2
    zeros = np.zeros((n_row ** 2, 1))
    c_varied = np.repeat(np.linspace(-1., 1., n_row)[:, np.newaxis], n_row, 0)
    c1 = Variable(FloatTensor(np.concatenate((c_varied, zeros, zeros, zeros, zeros, zeros, zeros, zeros), -1)))
    c2 = Variable(FloatTensor(np.concatenate((zeros, c_varied, c_varied, zeros, zeros, zeros, zeros, zeros), -1)))
    c3 = Variable(FloatTensor(np.concatenate((zeros, zeros, c_varied, zeros, zeros, zeros, zeros, zeros), -1)))
    c4 = Variable(FloatTensor(np.concatenate((zeros, zeros, zeros, c_varied, c_varied, zeros, zeros, zeros), -1)))
    c5 = Variable(FloatTensor(np.concatenate((zeros, zeros, zeros, zeros, c_varied, zeros, zeros, zeros), -1)))
    c6 = Variable(FloatTensor(np.concatenate((zeros, zeros, zeros, zeros, zeros, c_varied, zeros, zeros), -1)))
    c7 = Variable(FloatTensor(np.concatenate((zeros, zeros, zeros, zeros, zeros, zeros, c_varied, zeros), -1)))
    c8 = Variable(FloatTensor(np.concatenate((zeros, zeros, zeros, zeros, zeros, zeros, zeros, c_varied), -1)))
    sample1 = generator(static_z, static_label, c1)
    sample2 = generator(static_z, static_label, c2)
    sample3 = generator(static_z, static_label, c3)
    sample4 = generator(static_z, static_label, c4)
    sample5 = generator(static_z, static_label, c5)
    sample6 = generator(static_z, static_label, c6)
    sample7 = generator(static_z, static_label, c7)
    sample8 = generator(static_z, static_label, c8)


    grid_1 = make_grid(sample1.data, nrow =n_row)
    grid_2 = make_grid(sample2.data, nrow =n_row)
    grid_3 = make_grid(sample3.data, nrow =n_row)
    grid_4 = make_grid(sample4.data, nrow =n_row)
    grid_5 = make_grid(sample5.data, nrow =n_row)
    grid_6 = make_grid(sample6.data, nrow =n_row)
    grid_7 = make_grid(sample7.data, nrow =n_row)
    grid_8 = make_grid(sample8.data, nrow =n_row)

    save_image(grid_1, "images/varying_c1/%d.png" % batches_done, nrow=n_row, normalize=True)
    save_image(grid_2, "images/varying_c2/%d.png" % batches_done, nrow=n_row, normalize=True)
    save_image(grid_3, "images/varying_c3/%d.png" % batches_done, nrow=n_row, normalize=True)
    save_image(grid_4, "images/varying_c4/%d.png" % batches_done, nrow=n_row, normalize=True)
    save_image(grid_5, "images/varying_c5/%d.png" % batches_done, nrow=n_row, normalize=True)
    save_image(grid_6, "images/varying_c6/%d.png" % batches_done, nrow=n_row, normalize=True)
    save_image(grid_7, "images/varying_c7/%d.png" % batches_done, nrow=n_row, normalize=True)
    save_image(grid_8, "images/varying_c8/%d.png" % batches_done, nrow=n_row, normalize=True)

sample_image()

