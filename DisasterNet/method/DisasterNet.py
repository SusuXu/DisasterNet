import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.optim as optim
import numpy as np
import argparse
import time
import math
import random
import json
import pathlib
from torchinfo import summary
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from PIL import Image
import math
import scipy.io
import scipy.io as scio
import matplotlib.pyplot as plt
from model.model import DisasterNet
from flow.flow import Planar
from optimization.exp import run
from util.data_loader import tr_Dataset

parser = argparse.ArgumentParser(description='PyTorch Graphical VINF')

parser.add_argument('-d', '--dataset', type=str, default='mnist', choices=['mnist'],
                    metavar='DATASET',
                    help='Dataset choice.')

parser.add_argument('-nc', '--no_cuda', action='store_true', default=False,
                    help='disables CUDA training')

parser.add_argument('--manual_seed', type=int, help='manual seed, if not given resorts to random seed.')

parser.add_argument('-li', '--log_interval', type=int, default=10, metavar='LOG_INTERVAL',
                    help='how many batches to wait before logging training status')

parser.add_argument('-od', '--out_dir', type=str, default='logs/', metavar='OUT_DIR',
                    help='output directory for model snapshots etc.')

fp = parser.add_mutually_exclusive_group(required=False)
fp.add_argument('-te', '--testing', action='store_true', dest='testing',
                help='evaluate on test set after training')
fp.add_argument('-va', '--validation', action='store_false', dest='testing',
                help='only evaluate on validation set')
parser.set_defaults(testing=True)

# optimization settings
parser.add_argument('-e', '--epochs', type=int, default= 30, metavar='EPOCHS',
                    help='number of epochs to train (default: 30)')
parser.add_argument('-es', '--early_stopping_epochs', type=int, default=15, metavar='EARLY_STOPPING',
                    help='number of early stopping epochs')

parser.add_argument('-bs', '--batch_size', type=int, default=1000, metavar='BATCH_SIZE',
                    help='input batch size for training (default: 100)')
parser.add_argument('-lr', '--learning_rate', type=float, default=0.00001, metavar='LEARNING_RATE',
                    help='learning rate')

parser.add_argument('-a', '--anneal', type=str, default="std", choices= ["std", "off", "kl"], help="beta annealing scheme")
parser.add_argument('--max_beta', type=float, default=1., metavar='MB',
                    help='max beta for warm-up')
parser.add_argument('--min_beta', type=float, default=0.0, metavar='MB',
                    help='min beta for warm-up')
parser.add_argument('-f', '--flow', type=str, default='planar', choices=['planar', 'NICE', 'NICE_MLP', 'real' ])
parser.add_argument('-nf', '--num_flows', type=int, default=8,
                    metavar='NUM_FLOWS', help='Number of flow layers, ignored in absence of flows')
parser.add_argument('--z_size', type=int, default=1, metavar='ZSIZE',
                    help='how many stochastic hidden units')
parser.add_argument('--data_as_pseudo', type=bool, default=True, metavar='data_as_pseudo',
                    help='use random training data as pseudoinputs')

parser.add_argument('--gpu_num', type=int, default=0, metavar='GPU', help='choose GPU to run on.')

args = parser.parse_args(args=[])
args.cuda = not args.no_cuda and torch.cuda.is_available()
if args.manual_seed is None:
    args.manual_seed = 42
random.seed(args.manual_seed)
torch.manual_seed(args.manual_seed)
np.random.seed(args.manual_seed)
if args.cuda:
    torch.cuda.set_device(args.gpu_num)
args.data_as_pseudo = False

PLS = Image.open('/DisasterNet/DisasterNet/data/newLS.tif')
PLS = torch.from_numpy(np.array(PLS)).float()
a_LS = torch.from_numpy(np.array(PLS)).float()

PLF = Image.open('/DisasterNet/DisasterNet/data/newLF.tif')
PLF = torch.from_numpy(np.array(PLF)).float()
a_LF = torch.from_numpy(np.array(PLF)).float()

PBD = Image.open('/DisasterNet/DisasterNet/data/newBD.tif')
PBD = torch.from_numpy(np.array(PBD)).float()
a_BD = torch.from_numpy(np.array(PBD)).float()

DPM = Image.open('/DisasterNet/DisasterNet/data/DPM.tif')
DPM = torch.from_numpy(np.array(DPM)).float()
X = torch.from_numpy(np.array(DPM)).float()

q_LS_var = Image.open('/DisasterNet/DisasterNet/data/Haiti_LS_uncertainty1.tif')
q_LS_var = torch.from_numpy(np.array(q_LS_var)).float()

q_LF_var = Image.open('/DisasterNet/DisasterNet/data/Haiti_LF_uncertainty1.tif')
q_LF_var = torch.from_numpy(np.array(q_LF_var)).float()

BF = scipy.io.loadmat('/DisasterNet/DisasterNet/data/BD.mat')
BF = BF['BD']
BF = torch.from_numpy(np.array(BF)).float()

LOCAL = scipy.io.loadmat('/DisasterNet/DisasterNet/data/LOCAL.mat')
LOCAL = LOCAL['LOCAL']
LOCAL = torch.from_numpy(np.array(LOCAL)).float()

args.batch_size = 8109
PLS = PLS.view(-1,args.batch_size)
PLF = PLF.view(-1,args.batch_size)
PBD = PBD.view(-1,args.batch_size)

w_0LS = torch.tensor(0.2956)
w_0LF = torch.tensor(0.6414)
w_0BD = torch.tensor(0.2184)
w_eLS = torch.tensor(0.0186)
w_eLF = torch.tensor(0.9015)
w_eBD = torch.tensor(0.1920)
w_ex  = torch.tensor(0.0267)
w_0x  = torch.tensor(0.5337)
var_x = torch.tensor(0.1333)
w_LSx = torch.tensor(0.2258)
w_LFx = torch.tensor(0.6058)
w_BDx = torch.tensor(0.3021)

w_aBD  = torch.tensor(0.7886)
w_aLS  = torch.tensor(0.4105)
w_aLF  = torch.tensor(0.2108)

w_gBD = torch.tensor(0.8761) 
w_LSBD = torch.tensor(0.9303)
w_LFBD = torch.tensor(0.1363)
 
w_LSBD = torch.tensor(0.2513)
w_LFBD = torch.tensor(0.3612)
w_mu_LS = torch.rand(1, dtype=torch.float)
b_mu_LS = torch.rand(1, dtype=torch.float)
w_mu_LF = torch.rand(1, dtype=torch.float)
b_mu_LF = torch.rand(1, dtype=torch.float)
w_mu_BD = torch.rand(1, dtype=torch.float)
b_mu_BD = torch.rand(1, dtype=torch.float)

m = nn.Sigmoid()


# In[4]:


def reparameterize(mu, var):
    std = var.sqrt()
    eps = torch.randn_like(std)
    z = eps * std + mu
    z[z == 999] = 0
    z[z == 0] = 0
    return z
    # q_d \in [0,1]

def f(y):
    return torch.log(1-torch.exp(-y))

def flow_k(q, u, w, b):

    h = nn.Tanh()
    m = nn.Sigmoid()
    # Equation (10)
    q = q.unsqueeze(2)
    prod = torch.bmm(w, q) + b
    f_q = q + u * h(prod) # this is a 3d vector
    f_q = f_q.squeeze(2) # this is a 2d vector
    

    # compute logdetJ
    # Equation (11)
    psi = w * (1 - h(prod) ** 2)  # w * h'(prod)
    # Equation (12)
    log_det_jacobian = torch.log(torch.abs(1 + torch.bmm(psi, u)))
    log_det_jacobian = log_det_jacobian.squeeze(2).squeeze(1)

    return f_q, log_det_jacobian


mu = nn.Sequential(nn.Linear(args.batch_size, args.batch_size),
           nn.Hardtanh(min_val=-2, max_val=2))
var = nn.Sequential(nn.Linear(args.batch_size, args.batch_size),
                    nn.Softplus(),
                    nn.Hardtanh(min_val=1, max_val=5))
amor_u = nn.Sequential(nn.Linear(args.batch_size, args.num_flows*args.batch_size),
                       nn.Hardtanh(min_val=-1, max_val=1))
amor_w = nn.Sequential(nn.Linear(args.batch_size, args.num_flows*args.batch_size),
                       nn.Hardtanh(min_val=-1, max_val=1))
amor_b = nn.Sequential(nn.Linear(args.batch_size, args.num_flows),
                       nn.Hardtanh(min_val=-1, max_val=1))


# LS and LF part
PLS_1 = w_aLS*PLS + w_0LS
PLF_1 = w_aLF*PLF + w_0LF

q_LS_mu = w_mu_LS*PLS_1 + b_mu_LS
std_LS = q_LS_var.sqrt()
std_LS[q_LS_var == 999] = 0
eps_LS = torch.randn_like(std_LS)
q_LS = eps_LS * std_LS + q_LS_mu
q_LS[q_LS_var == 999] = 0
q_LS[q_LS_var == 0] = 0
q_LS[q_LS != 0] = m(q_LS[q_LS != 0])

q_LF_mu = w_mu_LF*PLF_1 + b_mu_LF
std_LF = q_LF_var.sqrt()
std_LF[q_LF_var == 999] = 0
eps_LF = torch.randn_like(std_LF)
q_LF = eps_LF * std_LF + q_LF_mu
q_LF[q_LF_var == 999] = 0
q_LF[q_LF_var == 0] = 0
q_LF[q_LF != 0] = m(q_LF[q_LF != 0])

q_LS = [q_LS]  
q_LF = [q_LF]

u_LS = amor_u(PLS_1).view(-1, args.num_flows, args.batch_size, 1)
w_LS = amor_w(PLS_1).view(-1, args.num_flows, 1, args.batch_size)
b_LS = amor_b(PLS_1).view(-1, args.num_flows, 1, 1)

u_LF = amor_u(PLF_1).view(-1, args.num_flows, args.batch_size, 1)
w_LF = amor_w(PLF_1).view(-1, args.num_flows, 1, args.batch_size)
b_LF = amor_b(PLF_1).view(-1, args.num_flows, 1, 1)

log_det_j_LS = 0.
log_det_j_LF = 0.
log_det_j_BD = 0.

for k in range(args.num_flows):
    q_LFk, log_det_jacobian_LF = flow_k(q_LF[k], u_LF[:, k, :, :], w_LF[:, k, :, :], b_LF[:, k, :, :])
    q_LFk[q_LF_var == 999] = 0
    q_LFk[q_LF_var == 0] = 0
    q_LFk[(LOCAL == 0)|(LOCAL == 1)|(LOCAL == 3)|(LOCAL == 5)] = 0
    q_LFk[q_LFk != 0] = (q_LFk[q_LFk != 0] - q_LFk[q_LFk != 0].min())/(q_LFk[q_LFk != 0].max() - q_LFk[q_LFk != 0].min())
    q_LF.append(q_LFk)
    log_det_j_LF = log_det_j_LF + log_det_jacobian_LF

for k in range(args.num_flows):
    q_LSk, log_det_jacobian_LS = flow_k(q_LS[k], u_LS[:, k, :, :], w_LS[:, k, :, :], b_LS[:, k, :, :])
    q_LSk[q_LS_var == 999] = 0
    q_LSk[q_LS_var == 0] = 0
    q_LSk[(LOCAL == 0)|(LOCAL == 2)|(LOCAL == 4)] = 0
    q_LSk[q_LSk != 0] = (q_LSk[q_LSk != 0] - q_LSk[q_LSk != 0].min())/(q_LSk[q_LSk != 0].max() - q_LSk[q_LSk != 0].min())
    q_LS.append(q_LSk)
    log_det_j_LS = log_det_j_LS + log_det_jacobian_LS

q_LF_0 = q_LF[0]
q_LS_0 = q_LS[0]
q_LF_K = q_LF[-1]
q_LS_K = q_LS[-1]

PBD_1 = w_LSBD*q_LS_K + w_LFBD*q_LF_K + w_aBD*PBD + w_0BD
q_BD_mu = w_mu_BD*PBD_1 + b_mu_BD
y_BD_var = var(PBD_1)
u_BD = amor_u(PBD_1).view(-1, args.num_flows, args.batch_size, 1)
w_BD = amor_w(PBD_1).view(-1, args.num_flows, 1, args.batch_size)
b_BD = amor_b(PBD_1).view(-1, args.num_flows, 1, 1)
q_BD_var = y_BD_var + w_eBD**2

q_BD_mu = w_mu_BD*PBD_1 + b_mu_BD
std_BD = q_BD_var.sqrt()
eps_BD = torch.randn_like(std_BD)
q_BD = eps_BD * std_BD + q_BD_mu
q_BD[BF == 0] = 0
q_BD[q_BD != 0] = (q_BD[q_BD != 0] - q_BD[q_BD != 0].min())/(q_BD[q_BD != 0].max() - q_BD[q_BD != 0].min())
q_BD = [q_BD]

for k in range(args.num_flows):
    q_BDk, log_det_jacobian_BD = flow_k(q_BD[k], u_BD[:, k, :, :], w_BD[:, k, :, :], b_BD[:, k, :, :])
    q_BDk[(LOCAL == 0)|(LOCAL == 1)|(LOCAL == 2)|(LOCAL == 5)] = 0
    q_BDk[q_BDk != 0] = (q_BDk[q_BDk != 0] - q_BDk[q_BDk != 0].min())/(q_BDk[q_BDk != 0].max() - q_BDk[q_BDk != 0].min())
    q_BD.append(q_BDk)
    log_det_j_BD = log_det_j_BD + log_det_jacobian_BD

q_BD_0 = q_BD[0]
q_BD_K = q_BD[-1]


scio.savemat('q_BD_K_1.mat',{'q_BD_K': q_BD_K.detach().numpy()})
scio.savemat('q_LS_K_1.mat',{'q_LS_K': q_LS_K.detach().numpy()})
scio.savemat('q_LF_K_1.mat',{'q_LF_K': q_LF_K.detach().numpy()})


# #### Experiment Setup
total_num = PLS.size(0)*PLS.size(1)
bnum = np.floor(total_num/args.batch_size)
rand_idx = np.array(random.sample(range(int(bnum)*args.batch_size),int(bnum)*args.batch_size))
IND = torch.from_numpy(rand_idx[0:int(bnum*args.batch_size)])
IND = IND.view(-1, args.batch_size)
IND = IND.numpy()

aLS = np.array(PLS).reshape([total_num])
aLS = aLS[rand_idx[0:int(bnum*args.batch_size)]]
aLS = torch.from_numpy(np.array(aLS)).float()
aLS = aLS.view(-1,args.batch_size)

aLF = np.array(PLF).reshape([total_num])
aLF = aLF[rand_idx[0:int(bnum*args.batch_size)]]
aLF = torch.from_numpy(np.array(aLF)).float()
aLF = aLF.view(-1,args.batch_size)

aBD = np.array(PBD).reshape([total_num])
aBD = aBD[rand_idx[0:int(bnum*args.batch_size)]]
aBD = torch.from_numpy(np.array(aBD)).float()
aBD = aBD.view(-1,args.batch_size)

obs = np.array(DPM).reshape([total_num])
obs = obs[rand_idx[0:int(bnum*args.batch_size)]]
obs = torch.from_numpy(np.array(obs)).float()
obs = obs.view(-1,args.batch_size)

var_LS = np.array(q_LS_var).reshape([total_num])
var_LS = var_LS[rand_idx[0:int(bnum*args.batch_size)]]
var_LS = torch.from_numpy(np.array(var_LS)).float()
var_LS = var_LS.view(-1,args.batch_size)

var_LF = np.array(q_LF_var).reshape([total_num])
var_LF = var_LF[rand_idx[0:int(bnum*args.batch_size)]]
var_LF = torch.from_numpy(np.array(var_LF)).float()
var_LF = var_LF.view(-1,args.batch_size)

FT = np.array(BF).reshape([total_num])
FT = FT[rand_idx[0:int(bnum*args.batch_size)]]
FT = torch.from_numpy(np.array(FT)).float()
FT = FT.view(-1,args.batch_size)

LOC = np.array(LOCAL).reshape([total_num])
LOC = LOC[rand_idx[0:int(bnum*args.batch_size)]]
LOC = torch.from_numpy(np.array(LOC)).float()
LOC = LOC.view(-1,args.batch_size)

LS = aLS.detach()
LF = aLF.detach()
BD = aBD.detach()
OB = obs.detach()
var_LS = var_LS.detach()
var_LF = var_LF.detach()
FT = FT.detach()
LOC = LOC.detach()

test_PLS = torch.tensor(PLS.view(total_num).numpy().tolist().copy()).view(-1,1)
test_PLF = torch.tensor(PLF.view(total_num).numpy().tolist().copy()).view(-1,1)
test_PBD = torch.tensor(PBD.view(total_num).numpy().tolist().copy()).view(-1,1)
dataset = tr_Dataset(LS, LF, BD, OB, var_LS, var_LF, FT, LOC)
MyDataLoader = DataLoader(dataset=dataset, shuffle=False)

model = DisasterNet(args)
if args.cuda:
    print("Model on GPU")
    model.cuda()
opt = optim.RMSprop(model.parameters(), lr=args.learning_rate, momentum=0.9)

test_PLS, test_PLF, test_PBD = run(MyDataLoader,model,opt,test_PLS,test_PLF,test_PBD,IND)
scio.savemat('final_QBD_flow_3.mat',{'final_QBD': test_PBD.view(PLS.size(0),-1).detach().numpy()})
scio.savemat('final_QLS_flow_3.mat',{'final_QLS': test_PLS.view(PLS.size(0),-1).detach().numpy()})
scio.savemat('final_QLF_flow_3.mat',{'final_QLF': test_PLF.view(PLS.size(0),-1).detach().numpy()})

