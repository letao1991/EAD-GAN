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
    code_input = torch.zeros((code_input_raw.shape[0], code_input_raw.shape[1]))

    code_input[:,0] = code_input_raw[:,0]* pq_factor + 1 #p
    code_input[:,1] = code_input_raw[:,1]* xy_factor #x
    code_input[:,2] = code_input_raw[:,2]* xy_factor #y

    return code_input

def from_latent_vector_2_color_para_pxy(code_input_raw):

    rgb_factor = 0.1
    code_input = torch.zeros((code_input_raw.shape[0], code_input_raw.shape[1]))

    code_input[:,0] = code_input_raw[:,0]* rgb_factor + 1#r
    code_input[:,1] = code_input_raw[:,1]* rgb_factor + 1#g
    code_input[:,2] = code_input_raw[:,2]* rgb_factor + 1#b

    return code_input


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


    return A_matrix















    
