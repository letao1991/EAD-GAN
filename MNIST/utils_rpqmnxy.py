import torch
from torch.autograd import Variable
import numpy as np
from scipy.optimize import minimize
from torch.optim import LBFGS
import torch.nn as nn

cuda = True if torch.cuda.is_available() else False
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


class Affine_classifier(nn.Module):
    def __init__(self):
        super(Affine_classifier, self).__init__()

        # Output layers
        self.fc_block = nn.Sequential(
            nn.Linear(6, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 7),
            )


    def forward(self, real_transfrom_code):

        cont = self.fc_block(real_transfrom_code)

        return cont

PATH = "rpqmnxy_approximator.pt"
BFGS_approximator = Affine_classifier()

BFGS_approximator.cuda()

BFGS_approximator.load_state_dict(torch.load(PATH))
BFGS_approximator.eval()
BFGS_approximator.requires_grad = False


def from_latent_vector_2_affine_para(code_input_raw):

    r_factor = 9
    pq_factor = 0.2
    mn_factor = 0.2
    xy_factor = 0.1

    code_input = torch.zeros((code_input_raw.shape[0], code_input_raw.shape[1])).cuda()

    code_input[:,0] = code_input_raw[:,0] * np.pi/r_factor #theta
    code_input[:,1] = code_input_raw[:,1] * pq_factor + 1 #p
    code_input[:,2] = code_input_raw[:,2] * pq_factor + 1 #q
    code_input[:,3] = code_input_raw[:,3]*mn_factor #m
    code_input[:,4] = code_input_raw[:,4]*mn_factor #n
    code_input[:,5] = code_input_raw[:,5]*xy_factor #x
    code_input[:,6] = code_input_raw[:,6]*xy_factor #y

    return code_input



def from_affine_para_2_latent_vector(affine_para):

    r_factor = 9
    pq_factor = 0.2
    mn_factor = 0.2
    xy_factor = 0.1

    code_rec = torch.zeros((affine_para.shape[0], affine_para.shape[1])).cuda()

    code_rec[:,0] = affine_para[:,0]/np.pi * r_factor #theta
    code_rec[:,1] = (affine_para[:,1] - 1)/pq_factor #p
    code_rec[:,2] = (affine_para[:,2] - 1)/pq_factor #q
    code_rec[:,3] = affine_para[:,3]/ mn_factor #x
    code_rec[:,4] = affine_para[:,4]/ mn_factor #y
    code_rec[:,5] = affine_para[:,5]/ xy_factor #x
    code_rec[:,6] = affine_para[:,6]/ xy_factor #y

    return code_rec


def get_matrix(code_input_raw):

    

    batch_size = code_input_raw.shape[0]

    code_input = from_latent_vector_2_affine_para(code_input_raw)

    rotation_matrix = torch.eye(3).unsqueeze(0).repeat(batch_size,1,1)
    zoom_matrix = torch.eye(3).unsqueeze(0).repeat(batch_size,1,1)
    trans_matrix = torch.eye(3).unsqueeze(0).repeat(batch_size,1,1)
    skew_matrix = torch.eye(3).unsqueeze(0).repeat(batch_size,1,1)

    rotation_matrix[:,0,0] = torch.cos(code_input[:,0])
    rotation_matrix[:,0,1] = -torch.sin(code_input[:,0])
    rotation_matrix[:,1,0] = torch.sin(code_input[:,0])
    rotation_matrix[:,1,1] = torch.cos(code_input[:,0])
    zoom_matrix[:,0,0] = code_input[:,1]
    zoom_matrix[:,1,1] = code_input[:,2]
    skew_matrix[:,0,1] = code_input[:,3]
    skew_matrix[:,1,0] = code_input[:,4]
    trans_matrix[:,0,2] = code_input[:,5]
    trans_matrix[:,1,2] = code_input[:,6]

    A_matrix = rotation_matrix @ zoom_matrix @ skew_matrix @ trans_matrix
    A_matrix = A_matrix.cuda()

    return A_matrix


def affine_regularizer(real_code, trans_code):

    real_code_affine = real_code
    trans_code_affine = trans_code

    # get affine matrix from latent vector
    real_matrix = get_matrix(real_code_affine)
    trans_matrix = get_matrix(trans_code_affine)

    relative_matrix = trans_matrix @ torch.inverse(real_matrix)

    relative_matrix_flat = torch.cat((relative_matrix[:,0], relative_matrix[:,1]), dim = 1)

    affine_code_pred = BFGS_approximator(relative_matrix_flat)

    code_rec_affine = from_affine_para_2_latent_vector(affine_code_pred)

    return code_rec_affine


