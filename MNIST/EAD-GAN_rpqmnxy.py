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
from utils_rpqmnxy import *


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



parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
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
print(opt)

cuda = True if torch.cuda.is_available() else False


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def to_categorical(y, num_columns):
    """Returns one-hot encoded Variable"""
    y_cat = np.zeros((y.shape[0], num_columns))
    y_cat[range(y.shape[0]), y] = 1.0

    return Variable(FloatTensor(y_cat))


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


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            """Returns layers of each discriminator block"""
            block = [spectral_norm(nn.Conv2d(in_filters, out_filters, 3, 2, 1)), nn.LeakyReLU(0.2, inplace=True)]
            #if bn:
                #block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.conv_blocks = nn.Sequential(
            *discriminator_block(opt.channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )
        #self.conv_blocks = nn.DataParallel(self.conv_blocks)

        # The height and width of downsampled image
        ds_size = opt.img_size // 2 ** 4

        # Output layers
        self.adv_layer = nn.Sequential(spectral_norm(nn.Linear(128 * ds_size ** 2, 1)))


    def forward(self, img):
        out_1 = self.conv_blocks(img)
        out_1 = out_1.view(out_1.shape[0], -1)
        #out = torch.cat((out_1, code), dim=1)
        validity = self.adv_layer(out_1)


        return validity


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        def encoder_block(in_filters, out_filters, bn=True):
            """Returns layers of each discriminator block"""
            block = [spectral_norm(nn.Conv2d(in_filters, out_filters, 3, 2, 1)), nn.LeakyReLU(0.2, inplace=True)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.conv_blocks = nn.Sequential(
            *encoder_block(opt.channels, 16, bn=False),
            *encoder_block(16, 32),
            *encoder_block(32, 64),
            *encoder_block(64, 128),
        )


        # The height and width of downsampled image
        ds_size = opt.img_size // 2 ** 4


        # Output layers
        self.aux_layer = nn.Sequential(spectral_norm(nn.Linear(128 * ds_size ** 2, opt.n_classes)), nn.Softmax())
        self.latent_layer = nn.Sequential(spectral_norm(nn.Linear(128 * ds_size ** 2, opt.code_dim)))
        self.noise_layer = nn.Sequential(spectral_norm(nn.Linear(128 * ds_size ** 2, opt.latent_dim)))



    def forward(self, img):
        out = self.conv_blocks(img)
        out = out.view(out.shape[0], -1)

        label = self.aux_layer(out)
        latent_code = self.latent_layer(out)
        noise = self.noise_layer(out)

        return label, latent_code, noise


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
adversarial_loss = torch.nn.MSELoss()
categorical_loss = torch.nn.CrossEntropyLoss()
continuous_loss = torch.nn.MSELoss()
affine_loss = torch.nn.MSELoss()

# Loss weights
lambda_cat = 1
lambda_con = 0.1
lambda_affine = 0.1

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()
encoder = Encoder()
trans_2D = transformation_2D()




if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")

if cuda:
    generator.cuda()
    discriminator.cuda()
    encoder.cuda()
    trans_2D.cuda()
    adversarial_loss.cuda()
    categorical_loss.cuda()
    continuous_loss.cuda()
    affine_loss.cuda()


# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)
encoder.apply(weights_init_normal)

# Configure data loader
os.makedirs("data/mnist", exist_ok=True)
dataloader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "data/mnist",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    ),
    batch_size=opt.batch_size,
    shuffle=True,
)

# Optimizers
optimizer_G = torch.optim.Adam([{'params' : generator.parameters()}], lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr*2, betas=(opt.b1, opt.b2))


optimizer_info = torch.optim.Adam(
    itertools.chain(generator.parameters(), encoder.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2)
)

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

# Static generator inputs for sampling
static_z = Variable(FloatTensor(np.zeros((opt.n_classes ** 2, opt.latent_dim))))
#static_label = to_categorical(
    #np.array([num for _ in range(opt.n_classes) for num in range(opt.n_classes)]), num_columns=opt.n_classes
#)
static_label = []
for i in range(opt.n_classes):
    for j in range(opt.n_classes):
        static_label.append(i)

static_label = to_categorical(np.asarray(static_label), num_columns=opt.n_classes)


static_code = Variable(FloatTensor(np.zeros((opt.n_classes ** 2, opt.code_dim))))


def sample_image(real_img, scaled_img, n_row, batches_done):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    # Static sample
    z = Variable(FloatTensor(np.random.normal(0, 1, (n_row ** 2, opt.latent_dim))))
    static_sample = generator(z, static_label, static_code)
    save_image(static_sample.data, "images/static/%d.png" % batches_done, nrow=n_row, normalize=True)

    # orginal image
    grid_real = make_grid(real_img.data, nrow =n_row)
    save_image(grid_real, "images/original/%d.png" % batches_done, nrow=n_row, normalize=True)

    # sacled image
    grid_scaled = make_grid(scaled_img.data, nrow =n_row)
    save_image(grid_scaled, "images/scaled/%d.png" % batches_done, nrow=n_row, normalize=True)




    # Get varied c1 and c2
    zeros = np.zeros((n_row ** 2, 1))
    #c_varied = np.repeat(np.linspace(-2, 2, n_row)[:, np.newaxis], n_row, 0)
    c_varied = np.tile(np.linspace(-2, 2, n_row), n_row)[:, np.newaxis]
    c1 = Variable(FloatTensor(np.concatenate((c_varied, zeros, zeros, zeros, zeros, zeros, zeros), -1)))
    c2 = Variable(FloatTensor(np.concatenate((zeros, c_varied, zeros, zeros, zeros, zeros, zeros), -1)))
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

    save_image(grid_1, "images/varying_c1/%d.png" % batches_done, nrow=n_row, normalize=True)
    save_image(grid_2, "images/varying_c2/%d.png" % batches_done, nrow=n_row, normalize=True)
    save_image(grid_3, "images/varying_c3/%d.png" % batches_done, nrow=n_row, normalize=True)
    save_image(grid_4, "images/varying_c4/%d.png" % batches_done, nrow=n_row, normalize=True)
    save_image(grid_5, "images/varying_c5/%d.png" % batches_done, nrow=n_row, normalize=True)
    save_image(grid_6, "images/varying_c6/%d.png" % batches_done, nrow=n_row, normalize=True)
    save_image(grid_7, "images/varying_c7/%d.png" % batches_done, nrow=n_row, normalize=True)


# ----------
#  Training
# ----------

for epoch in range(opt.n_epochs):
    for i, (imgs, labels) in enumerate(dataloader):

        batch_size = imgs.shape[0]

        # Adversarial ground truths
        valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
        fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(FloatTensor))
        labels = to_categorical(labels.numpy(), num_columns=opt.n_classes)


        sampled_labels = np.random.randint(0, opt.n_classes, batch_size)


        # Sample noise and labels as generator input
        z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
        label_input = to_categorical(sampled_labels, num_columns=opt.n_classes)
        code_input_array = np.random.uniform(-1, 1, (batch_size, opt.code_dim))
        code_input_original = Variable(FloatTensor(code_input_array))

        code_input = code_input_original.clone()

        #------------------------------
        # affine images

        A_matrix = get_matrix(code_input)

        scaled_img = trans_2D(real_imgs, A_matrix[:,0:2])

        #------------------------------

        # -----------------
        #  Train Generator + Encoder
        # -----------------

        optimizer_G.zero_grad()

        # Generate a batch of images
        gen_imgs = generator(z, label_input, code_input_original)

        #print (gen_code.shape)


        # Loss measures generator's ability to fool the discriminator
        validity= discriminator(gen_imgs)
        g_loss = adversarial_loss(validity, valid)

        g_loss.backward()
        optimizer_G.step()


        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Loss for real images
        real_pred = discriminator(scaled_img)
        d_real_loss = adversarial_loss(real_pred, valid)

        # Loss for fake images
        fake_pred = discriminator(gen_imgs.detach())
        d_fake_loss = adversarial_loss(fake_pred, fake)

        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        # ------------------
        # Information Loss + affine loss
        # ------------------

        optimizer_info.zero_grad()

        # Sample labels
        #sampled_labels = np.random.randint(0, opt.n_classes, batch_size)

        # Ground truth labels
        gt_labels = Variable(LongTensor(sampled_labels), requires_grad=False)


        gen_imgs = generator(z, label_input, code_input_original)
        pred_label, pred_code, _ = encoder(gen_imgs)

        info_loss_1 = lambda_cat * categorical_loss(pred_label, gt_labels) + lambda_con * continuous_loss(
            pred_code, code_input_original
        )

        transform_label, transform_code, _ = encoder(scaled_img)
        real_label, real_code, _ = encoder(real_imgs)


        predict_affine_numerical  = affine_regularizer(real_code, transform_code)



        affine_loss_cont = lambda_affine * continuous_loss(predict_affine_numerical, code_input_original)

        affine_loss = affine_loss_cont

        info_loss = info_loss_1 + affine_loss

        info_loss.backward()
        optimizer_info.step()

        # --------------
        # Log Progress
        # --------------
        batches_done = epoch * len(dataloader) + i

        if batches_done % 100 == 0:
            print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [info loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item(), info_loss.item())
            )
        
        if batches_done % opt.sample_interval == 0:
            sample_image(real_imgs[0:100], scaled_img[0:100], n_row=10, batches_done=batches_done)

        if batches_done % (opt.sample_interval * 10) == 0:


            torch.save(generator.state_dict(), "generator_%d.pt" % batches_done)
            torch.save(encoder.state_dict(), "encoder_%d.pt" % batches_done)

