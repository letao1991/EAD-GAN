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
img_shape = (64,64,1)


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


class DCIMetric():
    """ Impementation of the metric in: 
        A FRAMEWORK FOR THE QUANTITATIVE EVALUATION OF DISENTANGLED 
        REPRESENTATIONS
        Part of the code is adapted from:
        https://github.com/cianeastwood/qedr
    """
    def __init__(self, metric_data, regressor="Lasso", *args, **kwargs):
        super(DCIMetric, self).__init__(*args, **kwargs)
        self.data = metric_data["img_with_latent"]["img"]
        self.latents = metric_data["img_with_latent"]["latent"]

        self._regressor = regressor
        if regressor == "Lasso":
            self.regressor_class = Lasso
            self.alpha = 0.02
            # constant alpha for all models and targets
            self.params = {"alpha": self.alpha} 
            # weights
            self.importances_attr = "coef_" 
        elif regressor == "LassoCV":
            self.regressor_class = LassoCV
            # constant alpha for all models and targets
            self.params = {} 
            # weights
            self.importances_attr = "coef_" 
        elif regressor == "RandomForest":
            self.regressor_class = RandomForestRegressor
            # Create the parameter grid based on the results of random search 
            max_depths = [4, 5, 2, 5, 5]
            # Create the parameter grid based on the results of random search 
            self.params = [{"max_depth": max_depth, "oob_score": True}
                           for max_depth in max_depths]
            self.importances_attr = "feature_importances_"
        elif regressor == "RandomForestIBGAN":
            # The parameters that IBGAN paper uses
            self.regressor_class = RandomForestRegressor
            # Create the parameter grid based on the results of random search 
            max_depths = [4, 2, 4, 2, 2]
            # Create the parameter grid based on the results of random search 
            self.params = [{"max_depth": max_depth, "oob_score": True}
                           for max_depth in max_depths]
            self.importances_attr = "feature_importances_"
        elif regressor == "RandomForestCV":
            self.regressor_class = GridSearchCV
            # Create the parameter grid based on the results of random search 
            param_grid = {"max_depth": [i for i in range(2, 16)]}
            self.params = {
                "estimator": RandomForestRegressor(),
                "param_grid": param_grid,
                "cv": 3,
                "n_jobs": -1,
                "verbose": 0
            }
            self.importances_attr = "feature_importances_"
        elif "RandomForestEnum" in regressor:
            self.regressor_class = RandomForestRegressor
            # Create the parameter grid based on the results of random search 
            self.params = {
                "max_depth": int(regressor[len("RandomForestEnum"):]),
                "oob_score": True
            }
            self.importances_attr = "feature_importances_"
        else:
            raise NotImplementedError()

        self.TINY = 1e-12

    def normalize(self, X):
        mean = np.mean(X, 0) # training set
        stddev = np.std(X, 0) # training set
        #print('mean', mean)
        #print('std', stddev)
        return (X - mean) / stddev

    def norm_entropy(self, p):
        '''p: probabilities '''
        n = p.shape[0]
        return - p.dot(np.log(p + self.TINY) / np.log(n + self.TINY))

    def entropic_scores(self, r):
        '''r: relative importances '''
        r = np.abs(r)
        ps = r / np.sum(r, axis=0) # 'probabilities'
        hs = [1 - self.norm_entropy(p) for p in ps.T]
        return hs

    def evaluate(self):
        #codes = self.model.inference_from(self.data)

        encoder_pxy, encoder_r_cat = load_encoder()

        img = torch.from_numpy(self.data)
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
            

        # resize the image to the the same size and center position 
        cat, cont = encoder_r_cat(align_img)
        # move from torch tensor to numpy array
        cat = cat.cpu().detach().numpy()
        cont = cont.cpu().detach().numpy()
        align_code = align_code.cpu().detach().numpy()
        #print ("cont", cont.shape)
        #print ("cat", cat.shape)

        cat = np.argmax(cat, axis = 1).reshape(-1, 1)

        codes = np.concatenate((cat, cont[:,0:2], align_code[:, 1:3]), axis = 1)



        latents = self.latents
        
        codes = self.normalize(codes)
        latents = self.normalize(latents)
        R = []

        for j in range(self.latents.shape[-1]):
            if isinstance(self.params, dict):
              regressor = self.regressor_class(**self.params)
            elif isinstance(self.params, list):
              regressor = self.regressor_class(**self.params[j])
            regressor.fit(codes, latents[:, j])

            # extract relative importance of each code variable in 
            # predicting the latent z_j
            if self._regressor == "RandomForestCV":
                best_rf = regressor.best_estimator_
                r = getattr(best_rf, self.importances_attr)[:, None]
            else:
                r = getattr(regressor, self.importances_attr)[:, None]

            R.append(np.abs(r))

        R = np.hstack(R) #columnwise, predictions of each z

        # disentanglement
        disent_scores = self.entropic_scores(R.T)
        # relative importance of each code variable
        c_rel_importance = np.sum(R, 1) / np.sum(R) 
        disent_w_avg = np.sum(np.array(disent_scores) * c_rel_importance)

        # completeness
        complete_scores = self.entropic_scores(R)
        complete_avg = np.mean(complete_scores)

        print ("disent_scores", disent_scores)
        print ("complete_avg", complete_avg)

        return {
            "DCI_{}_disent_metric_detail".format(self._regressor): \
                disent_scores,
            "DCI_{}_disent_metric".format(self._regressor): disent_w_avg,
            "DCI_{}_complete_metric_detail".format(self._regressor): \
                complete_scores,
            "DCI_{}_complete_metric".format(self._regressor): complete_avg,
            "DCI_{}_metric_detail".format(self._regressor): R
            }



    

_, metric_data, _, _ = load_data()

SAP = DCIMetric(metric_data)
SAP.evaluate()
    










