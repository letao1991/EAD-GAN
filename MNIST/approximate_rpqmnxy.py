
import numpy as np
import torch
from torch.autograd import Variable

import torch.nn as nn

import time


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

affine_approx_loss = torch.nn.MSELoss()
affine_classifier = Affine_classifier()


if cuda:
	affine_classifier.cuda()
	affine_approx_loss.cuda()


optimizer_affine_approx = torch.optim.Adam(affine_classifier.parameters(), lr=0.0002, betas=(0.5, 0.999))


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



def get_matrix_rpqmnxy(code_input_raw):

	

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

    return A_matrix, code_input

	




if __name__ == "__main__":

	if torch.cuda.device_count() > 1:
  		print("Let's use", torch.cuda.device_count(), "GPUs!")

	#start_time = time.time()

	
	for iteration in range(20001):	 

		batch_size = 128


		code_input_array = (np.random.rand(batch_size, 7)-0.5)*2
		code_input_original = Variable(FloatTensor(code_input_array))
		code_input_raw = code_input_original.clone()

		A_matrix, code_input = get_matrix_rpqmnxy(code_input_raw)
		A_matrix_flat = torch.cat((A_matrix[:,0], A_matrix[:,1]), dim = 1)

		
		optimizer_affine_approx.zero_grad()


		affine_code_pred = affine_classifier(A_matrix_flat)
		affine_loss = affine_approx_loss(affine_code_pred, code_input)

		affine_loss.backward()
		optimizer_affine_approx.step()

		if iteration % 1000 == 0:

			print(
		    	"[iteration %d]  [affine_loss: %f] "
		    	% (iteration, affine_loss.item())
			)

			print ("affine_code_pred", affine_code_pred[0])
			print ("code_input_original", code_input[0])

		if iteration % 20000 == 0:
			PATH = "rpqmnxy_approximator.pt"
			torch.save(affine_classifier.state_dict(), PATH)
		



		









