import numpy as np
import sys
import sklearn
from sklearn import metrics
import pandas as pd
from sklearn.linear_model import LogisticRegression, Lasso, LassoCV
import torch.nn as nn
import torch.nn.functional as F
import torch
import os
import numpy as np
import math
import itertools
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
from torchvision.utils import make_grid
from torch.nn.utils import spectral_norm
from utils_pxy import *

os.makedirs("images/original/", exist_ok=True)


def load_data():
    dataset_zip = np.load( "../dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz", encoding = 'latin1',allow_pickle=True )
    imgs_raw = dataset_zip['imgs']
    latents_values = dataset_zip['latents_values']

    img = torch.from_numpy(imgs_raw)
    img = img.unsqueeze(1)
    img = img.float()


    return img, latents_values
    


code_dim = 7
n_classes = 3



class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.conv_block = nn.Sequential(
            # [-1, 3, 32, 32] -> [-1, 128, 16, 16]
            spectral_norm(nn.Conv2d(3, 32, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),

            # [-1, 256, 8, 8]
            spectral_norm(nn.Conv2d(32, 32, 4, 2, 1)),
            #nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # [-1, 512, 4, 4]
            spectral_norm( nn.Conv2d(32, 64, 4, 2, 1)),
            #nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm( nn.Conv2d(64, 64, 4, 2, 1)),
            #nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),

            # [-1, 1 + cc_dim + dc_dim, 1, 1]
            #nn.Conv2d(64, opt.code_dim , 3, 1, 0)
        )


        self.fc1 = nn.Sequential(spectral_norm(nn.Linear(1024, 128)), nn.LeakyReLU(0.2, inplace = True))
        self.fc2 = nn.Sequential(spectral_norm(nn.Linear(128, 128)), nn.LeakyReLU(0.2, inplace = True))
        self.cat_layer = nn.Sequential(spectral_norm(nn.Linear(128, n_classes)), nn.Softmax())
        self.cont_layer = nn.Sequential(spectral_norm(nn.Linear(128, code_dim)))

    def forward(self, img):
        x = self.conv_block(img)
        x = x.view(x.shape[0], -1)

        x = self.fc1(x)
        x = self.fc2(x)
        cat = self.cat_layer(x)
        cont = self.cont_layer(x)

        return cat, cont




class Encoder_pxy(nn.Module):
    def __init__(self):
        super(Encoder_pxy, self).__init__()

        self.conv_block = nn.Sequential(
            # [-1, 3, 32, 32] -> [-1, 128, 16, 16]
            (nn.Conv2d(3, 32, 4, 2, 1)),
            nn.LeakyReLU(0.1, inplace=True),

            # [-1, 256, 8, 8]
            (nn.Conv2d(32, 32, 4, 2, 1)),
            #nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),

            # [-1, 512, 4, 4]
            ( nn.Conv2d(32, 64, 4, 2, 1)),
            #nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),

            ( nn.Conv2d(64, 64, 4, 2, 1)),
            #nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),

            # [-1, 1 + cc_dim + dc_dim, 1, 1]
            #nn.Conv2d(64, opt.code_dim , 3, 1, 0)
        )

        self.fc1 = nn.Linear(1024, 6)

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

trans_2D = transformation_2D()

        
def load_encoder():
    encoder_pxy = Encoder_pxy()
    PATH_pxy = "encoder_pxy_color_50000.pt"
    encoder_pxy.load_state_dict(torch.load(PATH_pxy))
    encoder_pxy.eval()

    encoder_r_cat = Encoder()
    PATH = "encoder_500000.pt"
    encoder_r_cat.load_state_dict(torch.load(PATH))
    encoder_r_cat.eval()


    return encoder_pxy, encoder_r_cat

def sample_img(img_resize):
    
    img_resize = (img_resize - 0.5)*2
    grid_real = make_grid(img_resize.data, nrow = 4)
    save_image(grid_real, "images/original/%d.png" % 0, nrow= 4, normalize=True)


def add_color_2_img(img):

    #img =  img.repeat(1,3,1,1)

    color_code = np.random.uniform(0.5, 1, [img.shape[0], 3, 1, 1])
    color = np.repeat(
            np.repeat(
            #random_state.uniform(0.5, 1, [img.shape[0], 1, 1, 3]),
            color_code,
            img.shape[2],
            axis=2),
            img.shape[3],
            axis=3)

    img = img * torch.from_numpy(color)
    img = img.float()

    return img, color_code


def generate_batch_factor_code(imgs, latents_values, representation_function_pxy, representation_function_r_cat,
       num_points, batch_size):
    #Sample a single training sample based on a mini-batch of ground-truth data.

    representations = None
    factors = None
    i = 0
    #print ("num_points", num_points)
    #print ("batch_size", batch_size)
    while i < num_points:
        num_points_iter = min(num_points - i, batch_size)

        select_index = np.random.randint(imgs.shape[0], size = num_points_iter)
        current_observations = imgs[select_index]
        current_factors = latents_values[select_index]
        
        current_observations, color_factor = add_color_2_img(current_observations)


        #current_factors = np.concatenate((current_factors, color_factor.squeeze()), axis = 1)
        

        
        if i == 0:
            factors = current_factors

            align_code = encoder_pxy(current_observations)
            align_matrix = get_matrix_pxy_align(align_code)
            inv_align_matrix = torch.inverse(align_matrix)
            align_img_affine = trans_2D(current_observations, inv_align_matrix[:,0:2])

            align_color_para = from_latent_vector_2_color_para_pxy(align_code[:,3:])
            align_color_para = align_color_para.unsqueeze(2).unsqueeze(3)
            align_color_para = align_color_para.repeat(1,1,align_img_affine.shape[-2], align_img_affine.shape[-1])

            align_img = align_img_affine/align_color_para

            # resize the image to the the same size and center position 
            cat, cont = encoder_r_cat(align_img)

            # move from torch tensor to numpy array
            cat = cat.cpu().detach().numpy()
            cont = cont.cpu().detach().numpy()
            align_code = align_code.cpu().detach().numpy()
            

            cat = np.argmax(cat, axis = 1).reshape(-1, 1)

            representations = np.concatenate((cat, cont[:,0:2], align_code[:, 1:3]), axis = 1)


        else:
            factors = np.vstack((factors, current_factors))

            align_code = encoder_pxy(current_observations)
            align_matrix = get_matrix_pxy_align(align_code)
            inv_align_matrix = torch.inverse(align_matrix)
            align_img_affine = trans_2D(current_observations, inv_align_matrix[:,0:2])

            align_color_para = from_latent_vector_2_color_para_pxy(align_code[:,3:])
            align_color_para = align_color_para.unsqueeze(2).unsqueeze(3)
            align_color_para = align_color_para.repeat(1,1,align_img_affine.shape[-2], align_img_affine.shape[-1])

            align_img = align_img_affine/align_color_para

            # resize the image to the the same size and center position 
            cat, cont = encoder_r_cat(align_img)
            # move from torch tensor to numpy array
            cat = cat.cpu().detach().numpy()
            cont = cont.cpu().detach().numpy()
            align_code = align_code.cpu().detach().numpy()


            cat = np.argmax(cat, axis = 1).reshape(-1, 1)

            representation_new = np.concatenate((cat, cont[:,0:2], align_code[:, 1:3]), axis = 1)

            representations = np.vstack((representations, representation_new))

        i += num_points_iter
    return np.transpose(representations), np.transpose(factors)
    



def make_discretizer(target, num_bins):
    discretized = np.zeros_like(target)
    for i in range(target.shape[0]):
        discretized[i, :] = np.digitize(target[i, :], np.histogram(
            target[i, :], num_bins)[1][:-1])
    return discretized


def discrete_mutual_info(mus, ys):
    """Compute discrete mutual information."""
    num_codes = mus.shape[0] 
    num_factors = ys.shape[0]
    m = np.zeros([num_codes, num_factors])
    for i in range(num_codes): 
        for j in range(num_factors): 
            m[i, j] = metrics.mutual_info_score(ys[j, :], mus[i, :])
    return m


def discrete_entropy(ys):
    """Compute discrete mutual information."""
    num_factors = ys.shape[0] # 5
    h = np.zeros(num_factors)
    for j in range(num_factors):
        h[j] = metrics.mutual_info_score(ys[j, :], ys[j, :])
    return h


score_list = np.zeros((3,3))
imgs, latents_values = load_data()


for j in range (1):

    for i in range (1):

        latents_values_refined = latents_values[:,1:6]

        encoder_pxy, encoder_r_cat = load_encoder()
        mus_train, ys_train = generate_batch_factor_code(imgs, latents_values_refined, encoder_pxy, encoder_r_cat, 1000, 16)
        print (mus_train.shape)

        discretized_mus = make_discretizer(mus_train, 20)
        m = discrete_mutual_info(discretized_mus, ys_train)
        entropy = discrete_entropy(ys_train)
        sorted_m = np.sort(m, axis=0)[::-1]
        score_list[j,i] = np.mean(np.divide(sorted_m[0, :] - sorted_m[1, :], entropy[:]))
        print ("score_list", (i, score_list[j,i]))






