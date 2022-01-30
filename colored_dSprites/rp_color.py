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
from utils_rp_color import *


import torch.nn as nn
import torch.nn.functional as F
import torch



os.makedirs("images/original/", exist_ok=True)
os.makedirs("images/trans/", exist_ok=True)
os.makedirs("images/varying_c1/", exist_ok=True)
os.makedirs("images/varying_c2/", exist_ok=True)
os.makedirs("images/varying_c3/", exist_ok=True)
os.makedirs("images/varying_c4/", exist_ok=True)
os.makedirs("images/varying_c5/", exist_ok=True)
os.makedirs("images/varying_c6/", exist_ok=True)
os.makedirs("images/varying_c7/", exist_ok=True)



parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=128, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=200, help="dimensionality of the latent space")
parser.add_argument("--code_dim", type=int, default=7, help="latent code")
parser.add_argument("--n_classes", type=int, default=3, help="number of classes for dataset")
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

        self.fc1 = nn.Linear(1024, 6)

    def forward(self, img):
        x = self.conv_block(img)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)

        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv_block = nn.Sequential(
            spectral_norm(nn.Conv2d(3, 32, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(32, 32, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(32, 64, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(64, 64, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),

        )

        self.fc1 = nn.Sequential(spectral_norm(nn.Linear(1024, 128)), nn.LeakyReLU(0.2, inplace=True))
        self.fc2 = nn.Linear(128, 1)

    def forward(self, img):
        x = self.conv_block(img)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = F.sigmoid(x)

        return x



class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()


        self.conv_block = nn.Sequential(
            (nn.ConvTranspose2d(64, 64, 4, 2, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            (nn.ConvTranspose2d(64, 64, 4, 2, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            (nn.ConvTranspose2d(64, 64, 4, 2, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            (nn.ConvTranspose2d(64, 3, 4, 2, 1)),
            
        )

        self.fc1 = nn.Sequential(nn.Linear(opt.n_classes + opt.code_dim, 128), nn.ReLU()) 
        self.fc2 = nn.Sequential(nn.Linear(128, 64*4*4), nn.ReLU())

    def forward(self, z_c):
        x = self.fc1(z_c)
        x = self.fc2(x)

        x = x.view(x.shape[0], 64, 4,4)

        x = self.conv_block(x)
        x = F.sigmoid(x)

        return x


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.conv_block = nn.Sequential(
            spectral_norm(nn.Conv2d(3, 32, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(32, 32, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm( nn.Conv2d(32, 64, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm( nn.Conv2d(64, 64, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),

        )


        self.fc1 = nn.Sequential(spectral_norm(nn.Linear(1024, 128)), nn.LeakyReLU(0.2, inplace = True))
        self.fc2 = nn.Sequential(spectral_norm(nn.Linear(128, 128)), nn.LeakyReLU(0.2, inplace = True))
        self.cat_layer = nn.Sequential(spectral_norm(nn.Linear(128, opt.n_classes)), nn.Softmax())
        self.cont_layer = nn.Sequential(spectral_norm(nn.Linear(128, opt.code_dim)))

    def forward(self, img):
        x = self.conv_block(img)
        x = x.view(x.shape[0], -1)

        x = self.fc1(x)
        x = self.fc2(x)
        cat = self.cat_layer(x)
        cont = self.cont_layer(x)

        return cat, cont




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



def to_categorical(y, num_columns):
    """Returns one-hot encoded Variable"""
    y_cat = np.zeros((y.shape[0], num_columns))
    y_cat[range(y.shape[0]), y] = 1.0

    return Variable(FloatTensor(y_cat))


def mutual_info_loss(c_given_x, c):
        """The mutual information metric we aim to minimize"""
        eps = 1e-8

        conditional_entropy = torch.mean(- torch.sum(torch.log(c_given_x + eps) *c, dim=1))
        entropy = torch.mean(- torch.sum(torch.log(c + eps)*c, dim=1))

        return conditional_entropy + entropy



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
category_loss = torch.nn.CrossEntropyLoss()
adv_loss = torch.nn.BCELoss()

lambda_affine = 1

encoder_pxy = Encoder_pxy()
encoder = Encoder()
discriminator = Discriminator()
generator = Generator()
trans_2D = transformation_2D()

if cuda:
    encoder_pxy.cuda()
    encoder.cuda()
    discriminator.cuda()
    generator.cuda()
    trans_2D.cuda()
    continuous_loss.cuda()
    category_loss.cuda()
    adv_loss.cuda()

PATH_pxy = "encoder_pxy_color_50000.pt"
encoder_pxy.load_state_dict(torch.load(PATH_pxy))
encoder_pxy.eval()

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr= opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))


optimizer_info = torch.optim.Adam(
    itertools.chain(generator.parameters(), encoder.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2)
)

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor







def sample_image(real_imgs, trans_img_affine,  n_cols, batches_done):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    

    # orginal image
    real_imgs = (real_imgs - 0.5)*2
    grid_real = make_grid(real_imgs.data, nrow = n_cols)
    save_image(grid_real, "images/original/%d.png" % batches_done, nrow= n_cols, normalize=True)

    trans_img_affine = (trans_img_affine - 0.5)*2
    grid_trans = make_grid(trans_img_affine.data, nrow = n_cols)
    save_image(grid_trans, "images/trans/%d.png" % batches_done, nrow= n_cols, normalize=True)


    sampled_labels = ([0,1,2,0,1,2,0])
    sampled_labels = np.repeat(sampled_labels, n_cols)
    label_input = to_categorical(sampled_labels, num_columns = opt.n_classes)



    zeros = np.zeros((n_cols * opt.code_dim, 1))
    c_varied = np.tile(np.linspace(-1, 1, n_cols)[:, np.newaxis], (7,1))

    #print ()
    c1 = Variable(FloatTensor(np.concatenate((c_varied, zeros, zeros, zeros, zeros, zeros, zeros), -1)))
    c2 = Variable(FloatTensor(np.concatenate((zeros, c_varied, zeros, zeros, zeros, zeros, zeros), -1)))
    c3 = Variable(FloatTensor(np.concatenate((zeros, zeros, c_varied, zeros, zeros, zeros, zeros), -1)))
    c4 = Variable(FloatTensor(np.concatenate((zeros, zeros, zeros, c_varied, zeros, zeros, zeros), -1)))
    c5 = Variable(FloatTensor(np.concatenate((zeros, zeros, zeros, zeros, c_varied, zeros, zeros), -1)))
    c6 = Variable(FloatTensor(np.concatenate((zeros, zeros, zeros, zeros, zeros, c_varied, zeros), -1)))
    c7 = Variable(FloatTensor(np.concatenate((zeros, zeros, zeros, zeros, zeros, zeros, c_varied), -1)))

    c_z_1 = torch.cat((label_input, c1), dim = 1)
    c_z_2 = torch.cat((label_input, c2), dim = 1)
    c_z_3 = torch.cat((label_input, c3), dim = 1)
    c_z_4 = torch.cat((label_input, c4), dim = 1)
    c_z_5 = torch.cat((label_input, c5), dim = 1)
    c_z_6 = torch.cat((label_input, c6), dim = 1)
    c_z_7 = torch.cat((label_input, c7), dim = 1)

    sample1 = (generator(c_z_1) - 0.5)*2
    sample2 = (generator(c_z_2) - 0.5)*2
    sample3 = (generator(c_z_3) - 0.5)*2
    sample4 = (generator(c_z_4) - 0.5)*2
    sample5 = (generator(c_z_5) - 0.5)*2
    sample6 = (generator(c_z_6) - 0.5)*2
    sample7 = (generator(c_z_7) - 0.5)*2

    grid_1 = make_grid(sample1.data, nrow =n_cols)
    grid_2 = make_grid(sample2.data, nrow =n_cols)
    grid_3 = make_grid(sample3.data, nrow =n_cols)
    grid_4 = make_grid(sample4.data, nrow =n_cols)
    grid_5 = make_grid(sample5.data, nrow =n_cols)
    grid_6 = make_grid(sample6.data, nrow =n_cols)
    grid_7 = make_grid(sample7.data, nrow =n_cols)

    save_image(grid_1, "images/varying_c1/%d.png" % batches_done, nrow=n_cols, normalize=True)
    save_image(grid_2, "images/varying_c2/%d.png" % batches_done, nrow=n_cols, normalize=True)
    save_image(grid_3, "images/varying_c3/%d.png" % batches_done, nrow=n_cols, normalize=True)
    save_image(grid_4, "images/varying_c4/%d.png" % batches_done, nrow=n_cols, normalize=True)
    save_image(grid_5, "images/varying_c5/%d.png" % batches_done, nrow=n_cols, normalize=True)
    save_image(grid_6, "images/varying_c6/%d.png" % batches_done, nrow=n_cols, normalize=True)
    save_image(grid_7, "images/varying_c7/%d.png" % batches_done, nrow=n_cols, normalize=True)



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


        # get zoom and position aligned images 
        align_code = encoder_pxy(img)
        align_matrix = get_matrix_pxy_align(align_code)
        inv_align_matrix = torch.inverse(align_matrix)
        align_img_affine = trans_2D(img, inv_align_matrix[:,0:2])

        align_color_para = from_latent_vector_2_color_para_pxy(align_code[:,3:])
        align_color_para = align_color_para.unsqueeze(2).unsqueeze(3)
        align_color_para = align_color_para.repeat(1,1,align_img_affine.shape[-2], align_img_affine.shape[-1])

        align_img = align_img_affine/align_color_para


        # ---------------------
        #  Train D, G
        # --------------------

        valid = Variable(FloatTensor(batch_size,1).fill_(1.0), requires_grad=False)
        fake = Variable(FloatTensor(batch_size,1).fill_(0.0), requires_grad=False)


        # get input latent code 
        code_input_array = np.random.uniform(-1, 1, (batch_size, opt.code_dim))
        code_input_original = torch.from_numpy(code_input_array).cuda()
        code_input_original = code_input_original.float()
        code_input = code_input_original.clone()

        sampled_labels = np.random.randint(0, opt.n_classes, batch_size)
        label_input = to_categorical(sampled_labels, num_columns=opt.n_classes)
        gt_labels = Variable((label_input), requires_grad=False)

        # get affine transformed images
        trans_matrix = get_matrix(code_input[:,:4])
        trans_img_affine = trans_2D(align_img, trans_matrix[:,0:2])

        # get color transformed images
        trans_color_para = from_latent_vector_2_color_para(code_input[:,4:])
        trans_color_para = trans_color_para.unsqueeze(2).unsqueeze(3)
        trans_color_para = trans_color_para.repeat(1,1,align_img.shape[-2], align_img.shape[-1])

        trans_img_affine_color = trans_img_affine * trans_color_para


        # train G

        z_c = torch.cat((label_input, code_input), dim = 1)
        gen_imgs = generator(z_c)


        # train D

        d_real = discriminator(trans_img_affine_color)
        d_fake = discriminator(gen_imgs.detach())

        d_loss_real = adv_loss(d_real, valid)
        d_loss_fake = adv_loss(d_fake, fake)
        d_loss = (d_loss_fake + d_loss_real)/2

        optimizer_D.zero_grad()
        d_loss.backward()
        optimizer_D.step()


        # train info
        code_input_array = np.random.uniform(-1, 1, (batch_size, opt.code_dim))
        code_input_original = torch.from_numpy(code_input_array).cuda()
        code_input_original = code_input_original.float()
        code_input = code_input_original.clone()

        sampled_labels = np.random.randint(0, opt.n_classes, batch_size)
        label_input = to_categorical(sampled_labels, num_columns=opt.n_classes)
        z_c = torch.cat((label_input, code_input), dim = 1)
        gt_labels = Variable((label_input), requires_grad=False)

        
        gen_imgs = generator(z_c)
        rec_cat, rec_cont = encoder(gen_imgs)

        g_fake = discriminator(gen_imgs)
        g_loss = adv_loss(g_fake, valid)


        

        cat_loss = mutual_info_loss(rec_cat, gt_labels)
        cont_loss = continuous_loss(rec_cont, code_input_original)

        info_loss = cat_loss + cont_loss


        # get zoom and position aligned images 
        align_code = encoder_pxy(img)
        align_matrix = get_matrix_pxy_align(align_code)
        inv_align_matrix = torch.inverse(align_matrix)
        align_img_affine = trans_2D(img, inv_align_matrix[:,0:2])

        align_color_para = from_latent_vector_2_color_para_pxy(align_code[:,3:])
        align_color_para = align_color_para.unsqueeze(2).unsqueeze(3)
        align_color_para = align_color_para.repeat(1,1,align_img_affine.shape[-2], align_img_affine.shape[-1])

        align_img = align_img_affine/align_color_para

        #relative affine
        # get distorted images
        trans_matrix = get_matrix(code_input[:,:4])
        trans_img_affine = trans_2D(align_img, trans_matrix[:,0:2])

        # relative color
        trans_color_para = from_latent_vector_2_color_para(code_input[:,4:])
        trans_color_para = trans_color_para.unsqueeze(2).unsqueeze(3)
        trans_color_para = trans_color_para.repeat(1,1,align_img.shape[-2], align_img.shape[-1])

        trans_img_affine_color = trans_img_affine * trans_color_para

        align_cat, align_cont = encoder(align_img)
        trans_cat, trans_cont = encoder(trans_img_affine_color)


        # get affine and color loss

        relative_affine_color = affine_color_regularzier(align_cont, trans_cont)
        affine_color_loss = continuous_loss(relative_affine_color, code_input_original)


        align_cat_gt = Variable((align_cat), requires_grad=False)

        relative_cat_loss = mutual_info_loss(trans_cat, align_cat_gt)

        info_affine_color_loss = info_loss + affine_color_loss + relative_cat_loss + g_loss

        optimizer_info.zero_grad()
        info_affine_color_loss.backward()
        optimizer_info.step()

        batches_done = epoch * len(dataloader) + i
        
        # --------------
        # Log Progress
        # --------------
        if batches_done % 100 == 0:
            print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [info cat loss: %f] [info cont loss: %f] [affine_color loss: %f] [relative_cat_loss: %f] "
            % (epoch, opt.n_epochs, i, len(dataloader),  d_loss.item(), g_loss.item(), cat_loss.item(), cont_loss.item(),
             affine_color_loss.item(), relative_cat_loss.item())
            )

            print ("trans_img_affine_color max", torch.max(trans_img_affine_color))
            print ("gen_imgs max", torch.max(gen_imgs))
            print ("trans_img_affine_color min", torch.min(trans_img_affine_color))
            print ("gen_imgs min", torch.min(gen_imgs))


        if batches_done % (opt.sample_interval * 2) == 0:
            sample_image(align_img[0:100],trans_img_affine_color[0:100], n_cols=10, batches_done=batches_done)

        if batches_done % (opt.sample_interval * 50 ) == 0:
            torch.save(encoder.state_dict(), "encoder_%d.pt" % batches_done)
            torch.save(generator.state_dict(), "generator_%d.pt" % batches_done)
