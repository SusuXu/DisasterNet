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
import math
import scipy.io
import scipy.io as scio
from flow import Planar

class DisasterNet(nn.Module):
    def __init__(self, args):
        super(DisasterNet, self).__init__()

        # extract model settings from args
        self.batch_size = args.batch_size
        self.is_cuda = args.cuda
        self.log_det_j_LS = 0.
        self.log_det_j_LF = 0.
        self.log_det_j_BD = 0.
        self.num_pseudos = args.num_pseudos # for initialising pseudoinputs
        
        flowLS = Planar 
        flowLF = Planar
        flowBD = Planar
        self.num_flows = args.num_flows
        self.m = nn.Sigmoid()

        for k in range(self.num_flows):
            flow_k_LS = flowLS()
            self.add_module('flow_LS_' + str(k), flow_k_LS)
        for k in range(self.num_flows):
            flow_k_LF = flowLF()
            self.add_module('flow_LF_' + str(k), flow_k_LF)
        for k in range(self.num_flows):
            flow_k_BD = flowBD()
            self.add_module('flow_BD_' + str(k), flow_k_BD)
        
        self.mu = nn.Sequential(nn.Linear(self.batch_size, self.batch_size),
                                nn.Hardtanh(min_val=-2, max_val=2))
        self.var = nn.Sequential(
            nn.Linear(self.batch_size, self.batch_size),
            nn.Softplus(),
            nn.Hardtanh(min_val=1, max_val=5))
        self.amor_u = nn.Sequential(nn.Linear(self.batch_size, self.num_flows * self.batch_size),
                                    nn.Hardtanh(min_val=-1, max_val=1))
        self.amor_w = nn.Sequential(nn.Linear(self.batch_size, self.num_flows * self.batch_size),
                                    nn.Hardtanh(min_val=-1, max_val=1))
        self.amor_b = nn.Sequential(nn.Linear(self.batch_size, self.num_flows),
                                    nn.Hardtanh(min_val=-1, max_val=1))

        # Parameters Setup
        self.w_0LS = nn.Parameter(torch.rand(1, requires_grad=True, dtype=torch.float))
        self.w_0LF = nn.Parameter(torch.rand(1, requires_grad=True, dtype=torch.float))
        self.w_0BD = nn.Parameter(torch.rand(1, requires_grad=True, dtype=torch.float))
        self.w_eLS = nn.Parameter(torch.rand(1, requires_grad=True, dtype=torch.float))
        self.w_eLF = nn.Parameter(torch.rand(1, requires_grad=True, dtype=torch.float))
        self.w_eBD = nn.Parameter(torch.rand(1, requires_grad=True, dtype=torch.float))
        self.w_ex  = nn.Parameter(torch.rand(1, requires_grad=True, dtype=torch.float))
        self.w_0x  = nn.Parameter(torch.rand(1, requires_grad=True, dtype=torch.float))
        self.var_x = nn.Parameter(torch.rand(1, requires_grad=True, dtype=torch.float))
        self.w_LSx = nn.Parameter(torch.rand(1, requires_grad=True, dtype=torch.float)) 
        self.w_LFx = nn.Parameter(torch.rand(1, requires_grad=True, dtype=torch.float)) 
        self.w_BDx = nn.Parameter(torch.rand(1, requires_grad=True, dtype=torch.float)) 
        self.w_LSBD = nn.Parameter(torch.rand(1, requires_grad=True, dtype=torch.float)) 
        self.w_LFBD = nn.Parameter(torch.rand(1, requires_grad=True, dtype=torch.float)) 
        self.w_aBD  = nn.Parameter(torch.rand(1, requires_grad=True, dtype=torch.float))
        self.w_aLS  = nn.Parameter(torch.rand(1, requires_grad=True, dtype=torch.float))
        self.w_aLF  = nn.Parameter(torch.rand(1, requires_grad=True, dtype=torch.float))
        self.w_mu_LS = nn.Parameter(torch.rand(1, requires_grad=True, dtype=torch.float))
        self.b_mu_LS = nn.Parameter(torch.rand(1, requires_grad=True, dtype=torch.float))
        self.w_mu_LF = nn.Parameter(torch.rand(1, requires_grad=True, dtype=torch.float))
        self.b_mu_LF = nn.Parameter(torch.rand(1, requires_grad=True, dtype=torch.float))
        self.w_mu_BD = nn.Parameter(torch.rand(1, requires_grad=True, dtype=torch.float))
        self.b_mu_BD = nn.Parameter(torch.rand(1, requires_grad=True, dtype=torch.float))
        if args.cuda:
            self.FloatTensor = torch.cuda.FloatTensor
        else:
            self.FloatTensor = torch.FloatTensor
          

    def forward(self, a_LS, a_LF, a_BD, x, qvar_LS, qvar_LF, ft, local):      
        if self.is_cuda:
            self.log_det_j_LS = torch.zeros([x.shape[0]]).cuda()
            self.log_det_j_LF = torch.zeros([x.shape[0]]).cuda()
            self.log_det_j_BD = torch.zeros([x.shape[0]]).cuda()
        else:
            self.log_det_j_LS = torch.zeros([x.shape[0]])
            self.log_det_j_LF = torch.zeros([x.shape[0]])
            self.log_det_j_BD = torch.zeros([x.shape[0]])

        a_LS_1 = self.w_aLS*a_LS 
        a_LF_1 = self.w_aLF*a_LF 
        
        z_LS_mu = self.w_mu_LS*a_LS_1 
        std_LS = qvar_LS.sqrt()
        std_LS[qvar_LS == 999] = 0
        eps_LS = torch.randn_like(std_LS)
        z_LS = eps_LS * std_LS + z_LS_mu
        z_LS[qvar_LS == 999] = 0
        z_LS[qvar_LS == 0] = 0
        z_LS[z_LS != 0] = self.m(z_LS[z_LS != 0])
    
        z_LF_mu = self.w_mu_LF*a_LF_1 
        std_LF = qvar_LF.sqrt()
        std_LF[qvar_LF == 999] = 0
        eps_LF = torch.randn_like(std_LF)
        z_LF = eps_LF * std_LF + z_LF_mu
        z_LF[qvar_LF == 999] = 0
        z_LF[qvar_LF == 0] = 0
        z_LF[z_LF != 0] = self.m(z_LF[z_LF != 0])

        z_LS = [z_LS]  
        z_LF = [z_LF]

        u_LS = self.amor_u(a_LS_1).view(-1, args.num_flows, args.batch_size, 1)
        w_LS = self.amor_w(a_LS_1).view(-1, args.num_flows, 1, args.batch_size)
        b_LS = self.amor_b(a_LS_1).view(-1, args.num_flows, 1, 1)

        u_LF = self.amor_u(a_LF_1).view(-1, args.num_flows, args.batch_size, 1)
        w_LF = self.amor_w(a_LF_1).view(-1, args.num_flows, 1, args.batch_size)
        b_LF = self.amor_b(a_LF_1).view(-1, args.num_flows, 1, 1)

        for k in range(self.num_flows):
            flow_k_LF = getattr(self, 'flow_LF_' + str(k))
            z_LFk, log_det_jacobian_LF = flow_k_LF(z_LF[k], u_LF[:, k, :, :], w_LF[:, k, :, :], b_LF[:, k, :, :])
            z_LFk[qvar_LF == 999] = 0
            z_LFk[qvar_LF == 0] = 0
            z_LFk[(local == 0)|(local == 1)|(local == 3)|(local == 5)] = 0
            z_LF.append(z_LFk)
            self.log_det_j_LF = self.log_det_j_LF + log_det_jacobian_LF
            
        for k in range(self.num_flows):
            flow_k_LS = getattr(self, 'flow_LS_' + str(k))
            z_LSk, log_det_jacobian_LS = flow_k_LS(z_LS[k], u_LS[:, k, :, :], w_LS[:, k, :, :], b_LS[:, k, :, :])
            z_LSk[qvar_LS == 999] = 0
            z_LSk[qvar_LS == 0] = 0
            z_LSk[(local == 0)|(local == 2)|(local == 4)] = 0
            z_LS.append(z_LSk)
            self.log_det_j_LS = self.log_det_j_LS + log_det_jacobian_LS



        z_LS_0 = z_LS[0]
        z_LF_0 = z_LF[0]
        z_LS_K = z_LS[-1]
        z_LF_K = z_LF[-1]  
        
        z_LS_K[z_LS_K != 0] = self.m(z_LS_K[z_LS_K != 0])
        z_LF_K[z_LF_K != 0] = self.m(z_LF_K[z_LF_K != 0])


        
        a_BD_1 = self.w_LSBD*z_LS_K.view(-1,self.batch_size) + self.w_LFBD*z_LF_K.view(-1,self.batch_size) + self.w_aBD*a_BD 
        z_BD_mu = self.w_mu_BD*a_BD_1 
        y_BD_var = self.var(a_BD_1)
        u_BD = self.amor_u(a_BD_1).view(-1, args.num_flows, args.batch_size, 1)
        w_BD = self.amor_w(a_BD_1).view(-1, args.num_flows, 1, args.batch_size)
        b_BD = self.amor_b(a_BD_1).view(-1, args.num_flows, 1, 1)
        z_BD_var = y_BD_var 
        std_BD = z_BD_var.sqrt()
        eps_BD = torch.randn_like(std_BD)
        z_BD = eps_BD * std_BD + z_BD_mu
        z_BD[ft == 0] = 0
        z_BD[z_BD != 0] = self.m(z_BD[z_BD != 0])
        z_BD = [z_BD]

        for k in range(self.num_flows):
            flow_k_BD = getattr(self, 'flow_BD_' + str(k))
            z_BDk, log_det_jacobian_BD = flow_k_BD(z_BD[k], u_BD[:, k, :, :], w_BD[:, k, :, :], b_BD[:, k, :, :])
            z_BDk[(local == 0)|(local == 1)|(local == 2)|(local == 5)] = 0
            z_BD.append(z_BDk)
            self.log_det_j_BD = self.log_det_j_BD + log_det_jacobian_BD
        z_BD_0 = z_BD[0]
        z_BD_K = z_BD[-1]
        z_BD_K[z_BD_K!=0] = self.m(z_BD_K[z_BD_K!=0])


        
        BD_inner_1 = torch.exp(- self.w_LSBD - self.w_LFBD)*z_LS_K*z_LF_K + (1-z_LS_K)*(1-z_LF_K) + torch.exp(-self.w_LSBD)*z_LS_K*(1-z_LF_K)+ torch.exp(-self.w_LFBD)*z_LF_K*(1-z_LS_K)
        BD_inner_2 = torch.exp(self.w_LSBD + self.w_LFBD)*z_LS_K*z_LF_K + (1-z_LS_K)*(1-z_LF_K) + torch.exp(self.w_LSBD)*z_LS_K*(1-z_LF_K)+ torch.exp(self.w_LFBD)*z_LF_K*(1-z_LS_K)
        p_z_BD = (z_BD_K*(-torch.log(1+torch.exp(-(self.w_eBD**2)/2 - self.w_0BD - self.w_aBD*a_BD)*BD_inner_1))
                 +(1-z_BD_K)*(-torch.log(1+torch.exp((self.w_eBD**2)/2 + self.w_0BD + self.w_aBD*a_BD)*BD_inner_2)))
        p_z_LS = (z_LS_K*(-torch.log(1+torch.exp(-(self.w_eLS**2)/2 - self.w_0LS - self.w_aLS*a_LS)))
                + (1-z_LS_K)*(-torch.log(1+torch.exp((self.w_eLS**2)/2 + self.w_0LS + self.w_aLS*a_LS))))
        p_z_LF = (z_LF_K*(-torch.log(1+torch.exp(-(self.w_eLF**2)/2 - self.w_0LF - self.w_aLF*a_LF)))
                + (1-z_LF_K)*(-torch.log(1+torch.exp((self.w_eLF**2)/2 + self.w_0LF + self.w_aLF*a_LF))))
        
        logp_zk = torch.mean(p_z_BD + p_z_LS + p_z_LF)
    
        log_p_xz = (- torch.log(x) - torch.log(torch.abs(torch.sqrt(self.w_ex))) 
                   - (torch.log(x)**2)/(2*(self.w_ex**2))
                   + (torch.log(x)*(self.w_BDx*z_BD_K + self.w_LSx*z_LS_K + self.w_LFx*z_LF_K + self.w_0x))/(self.w_ex**2)
                   - ((self.w_BDx**2)*z_BD_K + (self.w_LSx**2)*z_LS_K + (self.w_LFx**2)*z_LF_K + self.w_ex**2 + self.w_0x**2)/(2*(self.w_ex**2))
                   - (self.w_BDx*self.w_LSx*z_BD_K*z_LS_K + self.w_BDx*self.w_LFx*z_BD_K*z_LF_K + self.w_LSx*self.w_LFx*z_LS_K*z_LF_K)/(self.w_ex**2)
                   - (self.w_BDx*self.w_0x*z_BD_K + self.w_LSx*self.w_0x*z_LS_K + self.w_LFx*self.w_0x*z_LF_K)/(self.w_ex)**2)
                    

        return z_LS_mu, qvar_LS, z_LS_0, z_LS_K, self.log_det_j_LS, z_LF_mu, qvar_LF, z_LF_0, z_LF_K, self.log_det_j_LF,z_BD_mu, z_BD_var, z_BD_0, z_BD_K, self.log_det_j_BD,log_p_xz,logp_zk

