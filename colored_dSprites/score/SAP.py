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

from collections import Counter
import scipy.stats
from sklearn.linear_model import LogisticRegression, Lasso, LassoCV
from sklearn.svm import LinearSVC
from sklearn.metrics import mutual_info_score, roc_auc_score
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble.forest import RandomForestRegressor
from sklearn.model_selection import GridSearchCV



def load_data():
    # part of the code is from:
    # https://github.com/deepmind/dsprites-dataset/blob/master/dsprites_reloading_example.ipynb
    dataset_zip = np.load("../dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz", encoding = 'latin1', allow_pickle=True)
    imgs = dataset_zip['imgs']
    latent_values = dataset_zip['latents_values']
    #latents_classes = dataset_zip['latents_classes']
    metadata = dataset_zip['metadata'][()]

    #imgs = imgs.reshape(737280, 64, 64, 1).astype(np.float)  # 0 ~ 1

    latents_names = metadata["latents_names"]
    latents_sizes = metadata["latents_sizes"]
    latents_possible_values = metadata["latents_possible_values"]
    latents_bases = np.concatenate(
        (latents_sizes[::-1].cumprod()[::-1][1:], np.array([1, ])))

    def latent_to_index(latents):
        return np.dot(latents, latents_bases).astype(int)

    def sample_latent(size=1):
        samples = np.zeros((size, latents_sizes.size))
        for lat_i, lat_size in enumerate(latents_sizes):
            samples[:, lat_i] = np.random.randint(lat_size, size=size)
        return samples

    metric_data_groups = []
    L = 100
    M = 500

    for i in range(M):
        fixed_latent_id = i % 5 + 1
        latents_sampled = sample_latent(size=L)
        latents_sampled[:, fixed_latent_id] = \
            np.random.randint(latents_sizes[fixed_latent_id], size=1)
        # print(latents_sampled[0:10])
        indices_sampled = latent_to_index(latents_sampled)
        imgs_sampled = imgs[indices_sampled]
        metric_data_groups.append(
            {"img": imgs_sampled,
             "label": fixed_latent_id - 1})

    selected_ids = np.random.permutation(range(imgs.shape[0]))
    selected_ids = selected_ids[0: int(imgs.shape[0] / 10)]
    metric_data_eval_std = imgs[selected_ids]

    random_latent_ids = sample_latent(size= int(imgs.shape[0] / 10))
    random_latent_ids = random_latent_ids.astype(np.int32)
    random_ids = latent_to_index(random_latent_ids)
    assert random_latent_ids.shape == (int(imgs.shape[0] / 10), 6)
    random_imgs = imgs[random_ids]

    random_latents = np.zeros((random_imgs.shape[0], 6))
    for i in range(6):
        random_latents[:, i] = \
            latents_possible_values[latents_names[i]][random_latent_ids[:, i]]

    assert np.all(random_latents[:, 0] == 1)
    assert np.min(random_latents[:, 1]) == 1
    assert np.max(random_latents[:, 1]) == 3

    random_latents = random_latents[:, 1:]
    random_latents[:, 0] -= 1.0

    metric_data_img_with_latent = {
        "img": random_imgs,
        "latent": random_latents,
        "latent_id": random_latent_ids[:, 1:],
        "is_continuous": [False, True, True, True, True]}

    metric_data = {
        "groups": metric_data_groups,
        "img_eval_std": metric_data_eval_std,
        "img_with_latent": metric_data_img_with_latent}

    return imgs, metric_data, latent_values, metadata



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

trans_2D = transformation_2D().cuda()

        
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


class SAPMetric():
    """ Impementation of the metric in: 
        VARIATIONAL INFERENCE OF DISENTANGLED LATENT CONCEPTS FROM UNLABELED 
        OBSERVATIONS
        Part of the code is adapted from:
        https://github.com/google-research/disentanglement_lib/blob/master/disentanglement_lib/evaluation/metrics/sap_score.py
    """
    def __init__(self, metric_data, *args, **kwargs):
        super(SAPMetric, self).__init__(*args, **kwargs)
        self.metric_data = metric_data

    def evaluate(self):
        #data_inference = self.model.inference_from(
            #self.metric_data["img_with_latent"]["img"])

        encoder_pxy, encoder_r_cat = load_encoder()

        img = torch.from_numpy(self.metric_data["img_with_latent"]["img"])
        img = img.unsqueeze(1)
        img = img.float()

        img, color_factor = add_color_2_img(img)

        align_code = encoder_pxy(img)
        align_matrix = get_matrix_pxy_align(align_code)
        inv_align_matrix = torch.inverse(align_matrix)

        align_img_affine = trans_2D(img, inv_align_matrix[:,0:2])

        align_color_para = from_latent_vector_2_color_para_pxy(align_code[:,3:])
        align_color_para = align_color_para.unsqueeze(2).unsqueeze(3)
        align_color_para = align_color_para.repeat(1,1,align_img_affine.shape[-2], align_img_affine.shape[-1])

        align_img = align_img_affine/align_color_para

        cat, cont = encoder_r_cat(align_img)

        # move from torch tensor to numpy array
        cat = cat.cpu().detach().numpy()
        cont = cont.cpu().detach().numpy()
        align_code = align_code.cpu().detach().numpy()

        cat = np.argmax(cat, axis = 1).reshape(-1, 1)


        data_inference = np.concatenate((cat, cont[:,0:2], align_code[:, 1:3]), axis = 1)


        data_gt_latents = self.metric_data["img_with_latent"]["latent"]
        factor_is_continuous = \
            self.metric_data["img_with_latent"]["is_continuous"]

        num_latents = data_inference.shape[1]
        num_factors = len(factor_is_continuous)

        score_matrix = np.zeros([num_latents, num_factors])
        for i in range(num_latents):
            for j in range(num_factors):
                inference_values = data_inference[:, i]
                gt_values = data_gt_latents[:, j]
                if factor_is_continuous[j]:
                    cov = np.cov(inference_values, gt_values, ddof=1)
                    assert np.all(np.asarray(list(cov.shape)) == 2)
                    cov_cov = cov[0, 1]**2
                    cov_sigmas_1 = cov[0, 0]
                    cov_sigmas_2 = cov[1, 1]
                    score_matrix[i, j] = cov_cov / cov_sigmas_1 / cov_sigmas_2
                else:
                    gt_values = gt_values.astype(np.int32)
                    classifier = LinearSVC(C=0.01, class_weight="balanced")
                    classifier.fit(inference_values[:, np.newaxis], gt_values)
                    pred = classifier.predict(inference_values[:, np.newaxis])
                    score_matrix[i, j] = np.mean(pred == gt_values)
        sorted_score_matrix = np.sort(score_matrix, axis=0)
        score = np.mean(sorted_score_matrix[-1, :] - 
                        sorted_score_matrix[-2, :])

        print ("score", score)

        return {"SAP_metric": score,
                "SAP_metric_detail": score_matrix}



    

_, metric_data, _, _ = load_data()

SAP = SAPMetric(metric_data)
SAP.evaluate()
    










