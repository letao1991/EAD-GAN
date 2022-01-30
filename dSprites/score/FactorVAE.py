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
    imgs = dataset_zip['imgs']
    latent_values = dataset_zip['latents_values']
    metadata = dataset_zip['metadata'][()]
    
    #imgs = imgs.reshape(737280, 64, 64, 1).astype(np.float)
    
    latents_sizes = metadata['latents_sizes']
    latents_bases = np.concatenate(
            (latents_sizes[::-1].cumprod()[::-1][1:], np.array([1, ])))
    
    
    
    def latent_to_index(latents):
        #print("latents", latents)
        #print ()
        return np.dot(latents, latents_bases).astype(int)
    
    def sample_latent(size=1):
        samples = np.zeros((size, latents_sizes.size))
        
        #print ("enumerate(latents_sizes)", enumerate(latents_sizes))
        for lat_i, lat_size in enumerate(latents_sizes):
            samples[:, lat_i] = np.random.randint(lat_size, size=size)
            print ("lat_i", lat_i)
            print ("lat_size", lat_size)
            print ("samples[:,lat_i].shape", samples[:,lat_i].shape)
        print ("latents_sizes.size", latents_sizes.size)
        return samples
    
    
    metric_data_groups = []
    L = 100
    M = 500
    
    
    for i in range(M):
        fixed_latent_id = i % 5 + 1
        latents_sampled = sample_latent(size=L)
        #print ("i", i)
        #print ("latents_sampled", latents_sampled)
        #print("latents_sampled[:, fixed_latent_id]", latents_sampled[:, fixed_latent_id])
        latents_sampled[:, fixed_latent_id] = \
            np.random.randint(latents_sizes[fixed_latent_id], size=1)
        #print("latents_sampled[:, fixed_latent_id]", latents_sampled[:, fixed_latent_id])
        #print(latents_sampled[0:10])
        #print("latents_sampled[:, fixed_latent_id]_2", latents_sampled[:, fixed_latent_id])
        #print ("fixed_latent_id", fixed_latent_id)
        #print ("latents_sampled_2", latents_sampled)
        indices_sampled = latent_to_index(latents_sampled)
        print ("latents_sampled.shape", latents_sampled.shape)
        print ("indices_sampled.shape", indices_sampled.shape)
        #print ("indices_sampled", indices_sampled)
        imgs_sampled = imgs[indices_sampled]
        metric_data_groups.append(
            {"img": imgs_sampled,
            "label": fixed_latent_id - 1})
    
    selected_ids = np.random.permutation(range(imgs.shape[0]))
    #print ("imgs.shape[0]", imgs.shape[0])
    #print ("selected_ids.shape", selected_ids.shape)
    #print ("selected_ids", selected_ids)
    selected_ids = selected_ids[0: int(imgs.shape[0] / 10)]
    #print ("selected_ids", selected_ids)
    #print ("selected_ids.shape", selected_ids.shape)
    metric_data_eval_std = imgs[selected_ids]
    #print ("metric_data_eval_std", metric_data_eval_std)

    print ("metric_data_groups",len(metric_data_groups))
    print ("imgs.shape", imgs.shape)
    print ("latent_values.shape", latent_values.shape)
    print ("latents_sizes", latents_sizes)
    print ("latents_bases", latents_bases)
    print ("latents_sizes[::-1]", latents_sizes[::-1])
    print ("latents_sizes[::-1].cumprod()", latents_sizes[::-1].cumprod())
    print ("latents_sizes[::-1].cumprod()[::-1]", latents_sizes[::-1].cumprod()[::-1])
    print ("latents_sizes[::-1].cumprod()[::-1][1:]", latents_sizes[::-1].cumprod()[::-1][1:])
    print ("selected_ids", selected_ids.shape)


    metric_data = {
        "groups": metric_data_groups,
        "img_eval_std": metric_data_eval_std}

    return imgs, metric_data, latent_values, metadata



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

trans_2D = transformation_2D().cuda()

        
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

class FactorVAEMetric():
    """ Impementation of the metric in: 
        Disentangling by Factorising
    """
    def __init__(self, metric_data, *args, **kwargs):
        super(FactorVAEMetric, self).__init__(*args, **kwargs)
        self.metric_data = metric_data

    def evaluate(self):

        encoder_pxy, encoder_r_cat = load_encoder()


        img = torch.from_numpy(self.metric_data["img_eval_std"])
        img = img.unsqueeze(1)
        x_eval_sample = img.float()

        align_code = encoder_pxy(x_eval_sample)
        align_matrix = get_matrix_pxy_align(align_code)
        inv_align_matrix = torch.inverse(align_matrix)

        align_img = trans_2D(x_eval_sample, inv_align_matrix[:,0:2])

        # resize the image to the the same size and center position 

        cat, cont = encoder_r_cat(align_img)
        #cat = np.argmax(c_z[ :,0:num_classes,], axis = 1).reshape(-1, 1)
        # move from torch tensor to numpy array
        cat = cat.cpu().detach().numpy()
        cont = cont.cpu().detach().numpy()
        align_code = align_code.cpu().detach().numpy()

        cat = np.argmax(cat, axis = 1).reshape(-1, 1)


        data_inference = np.concatenate((cat, cont[:,0:2], align_code[:, 1:3]), axis = 1)


        eval_std = np.std(data_inference, axis=0, keepdims=True)

        labels = set(data["label"] for data in self.metric_data["groups"])

        train_data = np.zeros((eval_std.shape[1], len(labels)))

        for data in self.metric_data["groups"]:

            #if data["label"] == 0:
                #continue

            #else:
                img = torch.from_numpy(data["img"])
                img = img.unsqueeze(1)
                x_eval_sample = img.float()

                align_code = encoder_pxy(x_eval_sample)
                align_matrix = get_matrix_pxy_align(align_code)
                inv_align_matrix = torch.inverse(align_matrix)

                align_img = trans_2D(x_eval_sample, inv_align_matrix[:,0:2])

                # resize the image to the the same size and center position 

                cat, cont = encoder_r_cat(align_img)

                # move from torch tensor to numpy array
                cat = cat.cpu().detach().numpy()
                cont = cont.cpu().detach().numpy()
                align_code = align_code.cpu().detach().numpy()

                cat = np.argmax(cat, axis = 1).reshape(-1, 1)


                data_inference = np.concatenate((cat, cont[:,0:2], align_code[:, 1:3]), axis = 1)


                data_inference /= eval_std
                data_std = np.std(data_inference, axis=0)
                predict = np.argmin(data_std)
                train_data[predict, data["label"]] += 1

        total_sample = np.sum(train_data)
        maxs = np.amax(train_data, axis=1)
        correct_sample = np.sum(maxs)

        correct_sample_revised = np.flip(np.sort(maxs), axis=0)
        correct_sample_revised = np.sum(
            correct_sample_revised[0: train_data.shape[1]])

        print ("score", float(correct_sample) / total_sample)

        return {"factorVAE_metric": float(correct_sample) / total_sample,
                "factorVAE_metric_revised": (float(correct_sample_revised) /
                                             total_sample),
                "factorVAE_metric_detail": train_data}



_, metric_data, _, _ = load_data()
    
Factor_score = FactorVAEMetric(metric_data)
Factor_score.evaluate()









