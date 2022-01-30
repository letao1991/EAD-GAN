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


def from_latent_vector_2_affine_para_D(code_input_raw):

    r_factor = 9
    pq_factor = 0.2
    xy_factor = 0.1
    code_input = torch.zeros((code_input_raw.shape[0], code_input_raw.shape[1])).cuda()

    code_input[:,0] = code_input_raw[:,0]* np.pi/r_factor#theta
    code_input[:,1] = code_input_raw[:,1]* pq_factor + 1 #p
    code_input[:,2] = code_input_raw[:,2]* xy_factor #x
    code_input[:,3] = code_input_raw[:,3]* xy_factor #y

    return code_input


def get_matrix_D(code_input_raw):

    batch_size = code_input_raw.shape[0]
    code_input = from_latent_vector_2_affine_para_D(code_input_raw)

    rotation_matrix = torch.eye(3).unsqueeze(0).repeat(batch_size,1,1)
    zoom_matrix = torch.eye(3).unsqueeze(0).repeat(batch_size,1,1)
    trans_matrix = torch.eye(3).unsqueeze(0).repeat(batch_size,1,1)

    rotation_matrix[:,0,0] = torch.cos(code_input[:,0])
    rotation_matrix[:,0,1] = -torch.sin(code_input[:,0])
    rotation_matrix[:,1,0] = torch.sin(code_input[:,0])
    rotation_matrix[:,1,1] = torch.cos(code_input[:,0])
    zoom_matrix[:,0,0] = code_input[:,1]
    zoom_matrix[:,1,1] = code_input[:,1]
    trans_matrix[:,0,2] = code_input[:,2]
    trans_matrix[:,1,2] = code_input[:,3]

    A_matrix = rotation_matrix @ zoom_matrix @ trans_matrix
    A_matrix = A_matrix.cuda()

    return A_matrix


def from_latent_vector_2_affine_para(code_input_raw):

    r_factor = 9
    pq_factor = 0.2
    xy_factor = 0.1
    code_input = torch.zeros((code_input_raw.shape[0], code_input_raw.shape[1])).cuda()

    code_input[:,0] = code_input_raw[:,0]* np.pi/r_factor#theta
    code_input[:,1] = code_input_raw[:,1]* pq_factor + 1 #p
    code_input[:,2] = code_input_raw[:,2]* xy_factor #x
    code_input[:,3] = code_input_raw[:,3]* xy_factor #y

    return code_input


def from_affine_para_2_latent_vector(affine_color_para):

    r_factor = 9
    pq_factor = 0.2
    xy_factor = 0.1

    code_rec = torch.zeros((affine_color_para.shape[0], affine_color_para.shape[1])).cuda()

    code_rec[:,0] = affine_color_para[:,0]/np.pi * r_factor #theta
    code_rec[:,1] = (affine_color_para[:,1] - 1)/pq_factor #p
    code_rec[:,2] = affine_color_para[:,2]/ xy_factor #x
    code_rec[:,3] = affine_color_para[:,3]/ xy_factor #y

    return code_rec



def get_matrix(code_input_raw):

    batch_size = code_input_raw.shape[0]
    code_input = from_latent_vector_2_affine_para(code_input_raw)

    rotation_matrix = torch.eye(3).unsqueeze(0).repeat(batch_size,1,1)
    zoom_matrix = torch.eye(3).unsqueeze(0).repeat(batch_size,1,1)
    trans_matrix = torch.eye(3).unsqueeze(0).repeat(batch_size,1,1)

    rotation_matrix[:,0,0] = torch.cos(code_input[:,0])
    rotation_matrix[:,0,1] = -torch.sin(code_input[:,0])
    rotation_matrix[:,1,0] = torch.sin(code_input[:,0])
    rotation_matrix[:,1,1] = torch.cos(code_input[:,0])
    zoom_matrix[:,0,0] = code_input[:,1]
    zoom_matrix[:,1,1] = code_input[:,1]
    trans_matrix[:,0,2] = code_input[:,2]
    trans_matrix[:,1,2] = code_input[:,3]

    A_matrix = rotation_matrix @ zoom_matrix @ trans_matrix
    A_matrix = A_matrix.cuda()

    return A_matrix

def affine_regularzier(real_code, trans_code):

    real_code_affine = real_code[:,:4] # first 4 code: theta, p, x, y
    trans_code_affine = trans_code[:,:4]

    # get affine matrix from latent vector
    real_matrix = get_matrix(real_code_affine)
    trans_matrix = get_matrix(trans_code_affine)

    relative_matrix = trans_matrix @ torch.inverse(real_matrix)

    # use LSE to get affine parameter from affine matrix
    rec_theta = torch.atan((relative_matrix[:,1,0] - relative_matrix[:,0,1])/(relative_matrix[:,0,0] + relative_matrix[:,1,1]))

    rec_p_1 = torch.cos(rec_theta) * (relative_matrix[:,0,0] + relative_matrix[:,1,1])
    rec_p_2 = torch.sin(rec_theta) * (relative_matrix[:,1,0] - relative_matrix[:,0,1])
    rec_p = 0.5*(rec_p_1 + rec_p_2)

    rec_x = (relative_matrix[:,0,2]* torch.cos(rec_theta) + relative_matrix[:,1,2] * torch.sin(rec_theta))/rec_p
    rec_y = (relative_matrix[:,1,2]* torch.cos(rec_theta) - relative_matrix[:,0,2] * torch.sin(rec_theta))/rec_p

    # normalize the affine parameter back to latent vector

    affine_para = torch.stack((rec_theta, rec_p, rec_x, rec_y), dim = 1)
    code_rec_affine = from_affine_para_2_latent_vector(affine_para)

    code_rec = code_rec_affine



    return  code_rec.float()











    
