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

from utils_rpqxy import *


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
parser.add_argument("--n_epochs", type=int, default=50, help="number of epochs of training")
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


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.main = nn.Sequential(
            spectral_norm( nn.Conv2d(3, 128, 4, 2, 1)),
            nn.LeakyReLU(0.1, inplace=True),

            spectral_norm(nn.Conv2d(128, 256, 4, 2, 1)),
            nn.LeakyReLU(0.1, inplace=True),

            spectral_norm( nn.Conv2d(256, 512, 4, 2, 1)),
            nn.LeakyReLU(0.1, inplace=True),

            spectral_norm( nn.Conv2d(512, 1024, 4, 2, 1)),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(1024, 1 + opt.n_classes + opt.code_dim , 4, 1, 0)
        )


    def forward(self, img):

        out = self.main(img).squeeze()

        validity = F.sigmoid(out[:, 0])

        cat = F.softmax(out[:, opt.code_dim + 1: opt.code_dim + 1 + opt.n_classes])

        cont = out[:, 1: opt.code_dim + 1]
        


        return cat, cont, validity





class transformation_2D(nn.Module):

    def __init__(self):
        super(transformation_2D, self).__init__()

    def stn(self, x, matrix_2D):

        grid = F.affine_grid(matrix_2D, x.size())
        x = F.grid_sample(x, grid, padding_mode = 'border')
        return x

    def forward(self, img, matrix_2D):
        out = self.stn(img, matrix_2D)

        return out

# Loss functions
adversarial_loss = torch.nn.BCELoss()
continuous_loss = torch.nn.MSELoss()
affine_loss = torch.nn.MSELoss()
categorical_loss = torch.nn.CrossEntropyLoss()

# Loss weights
lambda_cat = 1
lambda_con = 1
lambda_affine = 1

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()
trans_2D = transformation_2D()



if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")

if cuda:
    generator.cuda()
    discriminator.cuda()
    trans_2D.cuda()
    adversarial_loss.cuda()
    categorical_loss.cuda()
    continuous_loss.cuda()
    affine_loss.cuda()


# Configure data loader


transform = transforms.Compose([
            transforms.Resize(opt.img_size),
            transforms.CenterCrop(opt.img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),
                (0.5, 0.5, 0.5))])

dataset = datasets.ImageFolder('data', transform=transform)

dataloader = torch.utils.data.DataLoader(dataset, 
                                            batch_size=opt.batch_size, 
                                            shuffle=True, num_workers=8)



# Optimizers
optimizer_G = torch.optim.Adam( generator.parameters(), lr= 0.001, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr= 0.0002, betas=(opt.b1, opt.b2))


optimizer_info = torch.optim.Adam(
    itertools.chain(generator.parameters(), discriminator.parameters()), lr= 0.0002, betas=(opt.b1, opt.b2)
)

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

# Static generator inputs for sampling
static_z = Variable(FloatTensor(np.zeros((opt.n_classes ** 2, opt.latent_dim))))
static_label = to_categorical(
    np.array([num for _ in range(opt.n_classes) for num in range(opt.n_classes)]), num_columns=opt.n_classes
)
static_code = Variable(FloatTensor(np.zeros((opt.n_classes ** 2, opt.code_dim))))





def sample_image(real_img, scaled_img, n_row, batches_done):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    # Static sample
    z = Variable(FloatTensor(np.random.normal(0, 1, (n_row ** 2, opt.latent_dim))))
    static_sample = generator(z,static_label, static_code)
    save_image(static_sample.data, "images/static/%d.png" % batches_done, nrow=n_row, normalize=True)

    # orginal image
    grid_real = make_grid(real_img.data, nrow =n_row)
    save_image(grid_real, "images/original/%d.png" % batches_done, nrow=n_row, normalize=True)

    # sacled image
    grid_scaled = make_grid(scaled_img.data, nrow =n_row)
    save_image(grid_scaled, "images/scaled/%d.png" % batches_done, nrow=n_row, normalize=True)


    zeros = np.zeros((n_row ** 2, 1))
    c_varied = np.repeat(np.linspace(-1, 1, n_row)[:, np.newaxis], n_row, 0)
    c1 = Variable(FloatTensor(np.concatenate((c_varied, zeros, zeros, zeros, zeros, zeros, zeros, zeros), -1)))
    c2 = Variable(FloatTensor(np.concatenate((zeros, c_varied, zeros, zeros, zeros, zeros, zeros, zeros), -1)))
    c3 = Variable(FloatTensor(np.concatenate((zeros, zeros, c_varied, zeros, zeros, zeros, zeros, zeros), -1)))
    c4 = Variable(FloatTensor(np.concatenate((zeros, zeros, zeros, c_varied, zeros, zeros, zeros, zeros), -1)))
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



    


# ----------
#  Training
# ----------

for epoch in range(opt.n_epochs):
    for i, (imgs, labels) in enumerate(dataloader):

        batch_size = imgs.shape[0]


        valid = Variable(FloatTensor(batch_size).fill_(1.0), requires_grad=False)
        fake = Variable(FloatTensor(batch_size).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(FloatTensor))

        z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))


        # ------------------ scaled image

        code_input_array = np.random.uniform(-1, 1, (batch_size, opt.code_dim))
        code_input_original = Variable(FloatTensor(code_input_array))
        code_input = code_input_original.clone()

        sampled_labels = np.random.randint(0, opt.n_classes, batch_size)
        label_input = to_categorical(sampled_labels, num_columns=opt.n_classes)


        #------------------------------
        
        # affine images

        A_matrix = get_matrix(code_input[:,:5])

        scaled_img = trans_2D(real_imgs, A_matrix[:,0:2])


        # -----------------
        #  Train Generator + Encoder
        # -----------------

        optimizer_G.zero_grad()

        # Generate a batch of images
        gen_imgs = generator(z, label_input, code_input)

        _, _, validity = discriminator(gen_imgs)


        g_loss = adversarial_loss(validity, valid)

        g_loss.backward()
        optimizer_G.step()



        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        _, _, real_pred = discriminator(scaled_img)
        d_real_loss = adversarial_loss(real_pred, valid)

        # Loss for fake images
        _, _, fake_pred = discriminator(gen_imgs.detach())
        d_fake_loss = adversarial_loss(fake_pred, fake)

        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()




        #------------------------
        # info
        #------------------------

        optimizer_info.zero_grad()

        gt_labels = Variable(LongTensor(sampled_labels), requires_grad=False)


        gen_imgs = generator(z, label_input, code_input)
        pred_label, pred_code, _ = discriminator(gen_imgs)

        info_loss_1 = lambda_cat * categorical_loss(pred_label, gt_labels) + lambda_con * continuous_loss(
            pred_code, code_input_original)


        transform_label, transform_code, _ = discriminator(scaled_img)
        real_label, real_code, _ = discriminator(real_imgs)


        predict_affine_analytical  = affine_regularzier(real_code, transform_code)

        affine_loss_cont = lambda_affine * continuous_loss(predict_affine_analytical, code_input_original[:,:5])

        affine_loss = affine_loss_cont


        info_loss = info_loss_1 + affine_loss

        info_loss.backward()
        optimizer_info.step()

        batches_done = epoch * len(dataloader) + i
        if batches_done % 10 == 0:
            print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
            )

        
        if batches_done % opt.sample_interval == 0:
            sample_image(real_imgs[0:100], scaled_img[0:100], n_row=10, batches_done=batches_done)

        if batches_done % (opt.sample_interval * 15) == 0:
            PATH = "checkpoint_%d.tar" % batches_done

            torch.save({
            'discriminator_state_dict': discriminator.state_dict(),
            'generator_state_dict': generator.state_dict(),
            'epoch': epoch,
            'batches_done': batches_done,
            
            }, PATH)





