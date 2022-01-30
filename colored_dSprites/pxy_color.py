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

from utils_pxy import *


import torch.nn as nn
import torch.nn.functional as F
import torch


os.makedirs("images/original/", exist_ok=True)
os.makedirs("images/align/", exist_ok=True)




parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=10, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=128, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=200, help="dimensionality of the latent space")
parser.add_argument("--code_dim", type=int, default=6, help="latent code")
parser.add_argument("--n_classes", type=int, default=10, help="number of classes for dataset")
parser.add_argument("--img_size", type=int, default=64, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=1000, help="interval between image sampling")
opt = parser.parse_args()
print(opt)

cuda = True if torch.cuda.is_available() else False



class Encoder_pxy(nn.Module):
    def __init__(self):
        super(Encoder_pxy, self).__init__()

        self.conv_block = nn.Sequential(
            (nn.Conv2d(3, 32, 4, 2, 1)),
            nn.LeakyReLU(0.1, inplace=True),

            (nn.Conv2d(32, 32, 4, 2, 1)),
            nn.LeakyReLU(0.1, inplace=True),

            ( nn.Conv2d(32, 64, 4, 2, 1)),
            nn.LeakyReLU(0.1, inplace=True),

            ( nn.Conv2d(64, 64, 4, 2, 1)),
            nn.LeakyReLU(0.1, inplace=True),

        )

        self.fc1 = nn.Linear(1024, opt.code_dim)

    def forward(self, img):
        x = self.conv_block(img)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)

        return x




class transformation_2D(nn.Module):

    def __init__(self):
        super(transformation_2D, self).__init__()

    def stn(self, x, matrix_2D):

        grid = F.affine_grid(matrix_2D, x.size())
        x = F.grid_sample(x, grid, padding_mode = 'zeros')
        return x

    def forward(self, img, matrix_2D):
        out = self.stn(img, matrix_2D)

        return out




dataset_zip = np.load( "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz", encoding = 'bytes' )
x_train = dataset_zip['imgs']
x_train_tensor = torch.from_numpy(x_train)


dataloader = torch.utils.data.DataLoader(
    x_train_tensor,
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers = 8
)


continuous_loss = torch.nn.MSELoss()
lambda_affine = 1
encoder_pxy = Encoder_pxy()
trans_2D = transformation_2D()
if cuda:
    encoder_pxy.cuda()
    trans_2D.cuda()
    continuous_loss.cuda()

# Optimizers
optimizer_E = torch.optim.Adam(encoder_pxy.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))


def sample_image(real_imgs, n_row, batches_done):
    """Saves a grid of generated digits ranging from 0 to n_classes"""

    # orginal image
    img = real_imgs.clone()
    img = (img - 0.5)*2
    grid_real = make_grid(img.data, nrow =n_row)
    save_image(grid_real, "images/original/%d.png" % batches_done, nrow=n_row, normalize=True)

    real_code = encoder_pxy(real_imgs)

    A_matrix = get_matrix_pxy_align(real_code[:,:3])
    inv_matrix = torch.inverse(A_matrix)
    align_img_affine = trans_2D(real_imgs, inv_matrix[:,0:2])

    align_color_para = from_latent_vector_2_color_para_pxy(real_code[:,3:])
    align_color_para = align_color_para.unsqueeze(2).unsqueeze(3)
    align_color_para = align_color_para.repeat(1,1,align_img_affine.shape[-2], align_img_affine.shape[-1])

    print ("align_color_para", align_color_para.shape)
    align_img = align_img_affine/align_color_para
    align_img = (align_img - 0.5)*2



    # sacled image
    grid_align = make_grid(align_img.data, nrow =n_row)
    save_image(grid_align, "images/align/%d.png" % batches_done, nrow=n_row, normalize=True)


# ----------
#  Training
# ----------


for epoch in range(opt.n_epochs):
    for i, (img) in enumerate(dataloader):

        batch_size = img.shape[0]


        img = img.unsqueeze(1).cuda()

        img =  img.repeat(1,3,1,1)

        color = np.repeat(
            np.repeat(
            np.random.uniform(0.5, 1, [img.shape[0], 3, 1, 1]),
            img.shape[2],
            axis=2),
            img.shape[3],
            axis=3)

        img = img * torch.from_numpy(color).cuda()
        img = img.float()



        code_input_array = np.random.uniform(-1, 1, (batch_size, opt.code_dim))
        code_input_original = torch.from_numpy(code_input_array).cuda()
        code_input = code_input_original.float().clone()

        # ---------------------
        #  Train Encoder
        # --------------------

        real_code = encoder_pxy(img)

        A_matrix = get_matrix_pxy(code_input)
        trans_img_affine = trans_2D(img, A_matrix[:,0:2])


        trans_color_para = from_latent_vector_2_color_para_pxy(code_input[:,3:])
        trans_color_para = trans_color_para.unsqueeze(2).unsqueeze(3)
        trans_color_para = trans_color_para.repeat(1,1,trans_img_affine.shape[-2], trans_img_affine.shape[-1])

        trans_img_affine_color = trans_img_affine * trans_color_para



        trans_code = encoder_pxy(trans_img_affine_color)

        code_rec = affine_regularzier_pxy(real_code, trans_code)

        affine_loss = continuous_loss(code_rec, code_input)


        optimizer_E.zero_grad()
        affine_loss.backward()
        optimizer_E.step()

        batches_done = epoch * len(dataloader) + i
        
        # --------------
        # Log Progress
        # --------------
        if batches_done % 100 == 0:
            print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader),  affine_loss.item())
            )


        if batches_done % (opt.sample_interval) == 0:
            sample_image(img[0:100], n_row=10, batches_done=batches_done)

        if batches_done % (opt.sample_interval * 10 ) == 0:
            torch.save(encoder_pxy.state_dict(), "encoder_pxy_color_%d.pt" % batches_done)