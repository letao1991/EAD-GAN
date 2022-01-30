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



def load_data():
    dataset_zip = np.load( "../dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz", encoding = 'latin1',allow_pickle=True )
    imgs_raw = dataset_zip['imgs']
    latents_values = dataset_zip['latents_values']

    img = torch.from_numpy(imgs_raw)
    img = img.unsqueeze(1)
    img = img.float()


    return img, latents_values
    


code_dim = 4
n_classes = 3
img_shape = (64,64,1)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.conv_block = nn.Sequential(
            # [-1, 3, 32, 32] -> [-1, 128, 16, 16]
            spectral_norm(nn.Conv2d(1, 32, 4, 2, 1)),
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
            (nn.Conv2d(1, 32, 4, 2, 1)),
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

        self.fc1 = nn.Linear(1024, 3)

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
    PATH_pxy = "encoder_pxy_50000.pt"
    encoder_pxy.load_state_dict(torch.load(PATH_pxy))
    encoder_pxy.eval()

    encoder_r_cat = Encoder()
    PATH = "encoder_500000.pt"
    encoder_r_cat.load_state_dict(torch.load(PATH))
    encoder_r_cat.eval()


    return encoder_pxy, encoder_r_cat

def sample_img(img_resize):
    

                print ("img_resize", img_resize.shape)
                img_resize = img_resize.squeeze(1)
                img_resize = img_resize.unsqueeze(3)
                img_resize = img_resize.cpu().detach().numpy()


                r,c = 4, 4
                

                fig, axs = plt.subplots(r,c)
                cnt = 0
                for i in range(r):
                    for j in range(c):
                        axs[i,j].imshow(img_resize[cnt, :,:,0], cmap = 'gray')
                        axs[i,j].axis('off')
                        cnt += 1
                plt.show()
                plt.close()


def generate_batch_factor_code(imgs, latents_values, representation_function_pxy, representation_function_r_cat,
       num_points, batch_size):
    """Sample a single training sample based on a mini-batch of ground-truth data.

    Args:
    ground_truth_data: GroundTruthData to be sampled from.
    representation_function: Function that takes observation as input and
      outputs a representation.
    num_points: Number of points to sample.
    random_state: Numpy random state used for randomness.
    batch_size: Batchsize to sample points.

    Returns:
    representations: Codes (num_codes, num_points)-np array.
    factors: Factors generating the codes (num_factors, num_points)-np array.
    """
    representations = None
    factors = None
    i = 0
    #print ("num_points", num_points)
    #print ("batch_size", batch_size)
    while i < num_points:
        num_points_iter = min(num_points - i, batch_size)

        select_index = np.random.randint(imgs.shape[0], size = num_points_iter)
        #print ("select_index", select_index)
        current_observations = imgs[select_index]
        current_factors = latents_values[select_index]
        #current_observations = (current_observations.astype('float32') - 0.5)/0.5

        #print ("current_observations", current_observations.shape)
        #print ("current_factors", current_factors.shape)


        #print ("num_points_iter", num_points_iter) #16
        #print ("current_factors", current_factors.shape) # 16,5
        #print ("current_factors", current_factors) #random
        #print ("current_observations", current_observations.shape) # 16, 64, 64, 1
        if i == 0:
            factors = current_factors

            align_code = encoder_pxy(current_observations)
            align_matrix = get_matrix_pxy_align(align_code)
            inv_align_matrix = torch.inverse(align_matrix)
            #enlarge_matrix = get_enlarge_matrix(align_code)
            #align_enlarge_matrix  = inv_align_matrix @ enlarge_matrix
            align_img = trans_2D(current_observations, inv_align_matrix[:,0:2])
            

            # resize the image to the the same size and center position 
            cat, cont = encoder_r_cat(align_img)
            # move from torch tensor to numpy array
            cat = cat.cpu().detach().numpy()
            cont = cont.cpu().detach().numpy()
            align_code = align_code.cpu().detach().numpy()
            #print ("cont", cont.shape)
            #print ("cat", cat.shape)

            cat = np.argmax(cat, axis = 1).reshape(-1, 1)

            representations = np.concatenate((cat, cont[:,0:2], align_code[:, 1:3]), axis = 1)

            #representations_pxy, img_resize = representation_function_pxy.predict(current_observations)
            #representations_r_cat = representation_function_r_cat.predict(img_resize)
            #representations = np.concatenate((representations_r_cat[:,0:5], representations_pxy[:, 1:3]), axis = 1)
            #representations = np.concatenate((representations_r_cat, representations_pxy[:, 0:3]), axis = 1)

        else:
            factors = np.vstack((factors, current_factors))

            align_code = encoder_pxy(current_observations)
            align_matrix = get_matrix_pxy_align(align_code)
            inv_align_matrix = torch.inverse(align_matrix)
            #enlarge_matrix = get_enlarge_matrix(align_code)
            #align_enlarge_matrix  = inv_align_matrix @ enlarge_matrix
            align_img = trans_2D(current_observations, inv_align_matrix[:,0:2])

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
    num_codes = mus.shape[0] # 10
    num_factors = ys.shape[0] # 5
    m = np.zeros([num_codes, num_factors])
    for i in range(num_codes): # 10
        for j in range(num_factors): # 5
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






