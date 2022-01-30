# EAD-GAN

environment 

pytorch: 1.71
CUDA: 11.0
python: 3.72

Pretrained model: https://drive.google.com/drive/folders/1PgsXFUc7aHxn_jA3fCFTHq1VxVavKer2?usp=sharing


<br/><br/>

MNIST:
training data: it will be automatically downloaded to "MNIST/data" once you run "EAD-GAN_rpqmnxy.py"

train:
- first run "python approximate_rpqmnxy.py"
- next run "EAD-GAN_rpqmnxy.py"
- the generated images are stored in the "MNIST/images" folder

inference:
- "python generate_image.py", the generated images are stored in the "MNIST/images" folder




<br/><br/>

CelebA:
training data: 
- download from: https://drive.google.com/drive/folders/0B7EVK8r0v71pTUZsaXdaSnZBZzg?resourcekey=0-rJlzl934LzC-Xp28GeIBzQ
- extract under the folder "CelebA/data"

train:
- "python EAD-GAN_celebA.py"
- the generated images are stored in the "CelebA/images" folder

inference:
- "python gen_imgs", the generated images are stored in the "CelebA/images" folder




<br/><br/>

dSprites:
training data:
- download from: https://github.com/deepmind/dsprites-dataset/blob/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz
- put the downloaded npz under "dSprites"

train: 
- first run "pxy.py"
- next run "rp.py"
- the generated images are stored in the "dSprites/images" folder

disentanglement score
- first put the model "encoder_pxy_50000.pt" and "encoder_500000.pt" in the "dSprites/score" folder
- to calculate a specific score, e.g., "python BetVAE.py"



<br/><br/>

colored-dSprites:
training data:
- download from: https://github.com/deepmind/dsprites-dataset/blob/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz
- put the downloaded npz under "colored_dSprites"

train: 
- first run "pxy_color.py"
- next run "rp_color.py"
- the generated images are stored in the "colored_dSprites/images" folder

disentanglement score
- first put the model "encoder_pxy_color_50000.pt" and "encoder_500000.pt" in the "colored_dSprites/score" folder
- to calculate a specific score, e.g., "python BetVAE.py"
