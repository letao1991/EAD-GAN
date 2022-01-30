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



def from_latent_vector_2_affine_para_pxy(code_input_raw):

    pq_factor = 0.1
    xy_factor = 0.1
    code_input = torch.zeros((code_input_raw.shape[0], code_input_raw.shape[1])).cuda()

    code_input[:,0] = code_input_raw[:,0]* pq_factor + 1 #p
    code_input[:,1] = code_input_raw[:,1]* xy_factor #x
    code_input[:,2] = code_input_raw[:,2]* xy_factor #y

    return code_input

def from_affine_para_2_latent_vector_pxy(affine_para):

    pq_factor = 0.1
    xy_factor = 0.1
    code_rec = torch.zeros((affine_para.shape[0], affine_para.shape[1])).cuda()

    code_rec[:,0] = (affine_para[:,0] - 1)/pq_factor #p
    code_rec[:,1] = affine_para[:,1]/ xy_factor #x
    code_rec[:,2] = affine_para[:,2]/ xy_factor #y

    return code_rec

def from_latent_vector_2_color_para_pxy(code_input_raw):

    rgb_factor = 0.1
    code_input = torch.zeros((code_input_raw.shape[0], code_input_raw.shape[1])).cuda()

    code_input[:,0] = code_input_raw[:,0]* rgb_factor + 1#r
    code_input[:,1] = code_input_raw[:,1]* rgb_factor + 1#g
    code_input[:,2] = code_input_raw[:,2]* rgb_factor + 1#b

    return code_input


def from_color_para_2_latent_vector_pxy(affine_color_para):

    rgb_factor = 0.1
    code_rec = torch.zeros((affine_color_para.shape[0], affine_color_para.shape[1])).cuda()

    code_rec[:,0] = (affine_color_para[:,0] - 1)/rgb_factor #r
    code_rec[:,1] = (affine_color_para[:,1] - 1)/rgb_factor #g
    code_rec[:,2] = (affine_color_para[:,2] - 1)/rgb_factor #b

    return code_rec


def get_matrix_pxy(code_input_raw):


    batch_size = code_input_raw.shape[0]
    code_input = from_latent_vector_2_affine_para_pxy(code_input_raw)

    zoom_matrix = torch.eye(3).unsqueeze(0).repeat(batch_size,1,1)
    trans_matrix = torch.eye(3).unsqueeze(0).repeat(batch_size,1,1)

    zoom_matrix[:,0,0] = code_input[:,0]
    zoom_matrix[:,1,1] = code_input[:,0]
    trans_matrix[:,0,2] = code_input[:,1]
    trans_matrix[:,1,2] = code_input[:,2]

    A_matrix = zoom_matrix @ trans_matrix
    A_matrix = A_matrix.cuda()

    return A_matrix


def get_matrix_pxy_align(code_input_raw):


    batch_size = code_input_raw.shape[0]
    code_input = from_latent_vector_2_affine_para_pxy(code_input_raw)

    zoom_matrix = torch.eye(3).unsqueeze(0).repeat(batch_size,1,1)
    trans_matrix = torch.eye(3).unsqueeze(0).repeat(batch_size,1,1)

    zoom_matrix[:,0,0] = code_input[:,0]
    zoom_matrix[:,1,1] = code_input[:,0]
    trans_matrix[:,0,2] = code_input[:,1]
    trans_matrix[:,1,2] = code_input[:,2]

    #A_matrix = zoom_matrix @ trans_matrix
    A_matrix = trans_matrix
    A_matrix = A_matrix.cuda()

    return A_matrix

def get_matrix_pxy_align_pos_size(code_input_raw):


    batch_size = code_input_raw.shape[0]
    code_input = from_latent_vector_2_affine_para_pxy(code_input_raw)

    zoom_matrix = torch.eye(3).unsqueeze(0).repeat(batch_size,1,1)
    trans_matrix = torch.eye(3).unsqueeze(0).repeat(batch_size,1,1)

    zoom_matrix[:,0,0] = code_input[:,0]
    zoom_matrix[:,1,1] = code_input[:,0]
    trans_matrix[:,0,2] = code_input[:,1]
    trans_matrix[:,1,2] = code_input[:,2]

    A_matrix = zoom_matrix @ trans_matrix
    #A_matrix = trans_matrix
    A_matrix = A_matrix.cuda()

    return A_matrix

    

def get_enlarge_matrix(code_input_raw):


    batch_size = code_input_raw.shape[0]

    enlarge_matrix = torch.eye(3).unsqueeze(0).repeat(batch_size,1,1)

    enlarge_matrix[:,0,0] = 0.6
    enlarge_matrix[:,1,1] = 0.6


    A_matrix = enlarge_matrix
    A_matrix = A_matrix.cuda()

    return A_matrix

def affine_regularzier_pxy(real_code, trans_code):

	# get affine matrix from latent vector
    real_matrix = get_matrix_pxy(real_code[:,:3])
    trans_matrix = get_matrix_pxy(trans_code[:,:3])

    relative_matrix = trans_matrix @ torch.inverse(real_matrix)

	# use LSE to get affine parameter from affine matrix
    rec_p = (relative_matrix[:,0,0] + relative_matrix[:, 1,1])/2
    rec_x = relative_matrix[:,0,2]/rec_p
    rec_y = relative_matrix[:,1,2]/rec_p

    # normalize the affine parameter back to latent vector
    affine_para = torch.stack((rec_p, rec_x, rec_y), dim = 1)
    code_rec_affine = from_affine_para_2_latent_vector_pxy(affine_para)

    real_para_color = from_latent_vector_2_color_para_pxy(real_code[:,3:])
    trans_para_color = from_latent_vector_2_color_para_pxy(trans_code[:,3:])

    relative_color = trans_para_color/real_para_color

    code_rec_color = from_color_para_2_latent_vector_pxy(relative_color)
    code_rec = torch.cat((code_rec_affine, code_rec_color), dim = 1)

    return  code_rec.float()











    
