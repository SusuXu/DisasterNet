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


def run(MyDataLoader,model,opt,test_PLS,test_PLF,test_PBD,IND):
    loss = []
    t_loss = 1e-6
    TLoss = []
    epoch = 0
    for epoch in range(1, args.epochs + 1):
        train_loss = np.zeros(len(MyDataLoader))
        index = -1
        i = 0
        batch_size = args.batch_size
        cuda = args.cuda
        for aLS, aLF, aBD, obs, z_var_LS, z_var_LF, bdft, loc in MyDataLoader: 
            
            index += 1
            beta = min(0.01 + ((epoch - 1) * (len(MyDataLoader)/args.batch_size) + index + 1) / args.epochs , 1)
            if args.cuda:
                aLS = aLS.cuda()
                aLF = aLF.cuda()
                aBD = aBD.cuda()
                obs = obs.cuda()
                z_var_LS = z_var_LS.cuda()
                z_var_LF = z_var_LF.cuda()
                bdft = bdft.cuda()
                loc = loc.cuda()
                
            aLS = aLS.view(-1,args.batch_size)
            aLF = aLF.view(-1,args.batch_size)
            aBD = aBD.view(-1,args.batch_size)
            obs = obs.view(-1,args.batch_size)
            z_var_LS = z_var_LS.view(-1,args.batch_size)
            z_var_LF = z_var_LF.view(-1,args.batch_size)
            bdft = bdft.view(-1,args.batch_size)
            loc = loc.view(-1,args.batch_size)

            q_LS_mu, q_LS_var, q_LS_0, q_LS_K, log_det_j_LS, q_LF_mu, q_LF_var, q_LF_0, q_LF_K, log_det_j_LF, q_BD_mu, q_BD_var, q_BD_0, q_BD_K, log_det_j_BD, log_p_xz, logp_zk = model(aLS, aLF, aBD, obs, z_var_LS, z_var_LF, bdft, loc)    
            new_q_LS_K = q_LS_K.view(-1,1)
            new_q_LF_K = q_LF_K.view(-1,1)
            new_q_BD_K = q_BD_K.view(-1,1)
            test_PLS[IND[i]] = new_q_LS_K
            test_PLF[IND[i]] = new_q_LF_K
            test_PBD[IND[i]] = new_q_BD_K
            log_q_z0_LS = log_normal_dist(q_LS_0, mean=q_LS_mu, logvar=q_LS_var.log(), dim=1)
            log_q_z0_LF = log_normal_dist(q_LF_0, mean=q_LF_mu, logvar=q_LF_var.log(), dim=1)
            log_q_z0_BD = log_normal_dist(q_BD_0, mean=q_BD_mu, logvar=q_BD_var.log(), dim=1)
            log_q_z0 = (log_q_z0_LS + log_q_z0_LF + log_q_z0_BD)/args.batch_size
            log_det_jacobians = log_det_j_LS + log_det_j_LF + log_det_j_BD
            triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
            penalty_LS = triplet_loss(q_LS_K, aLS, aLF)
            penalty_LF = triplet_loss(q_LF_K, aLF, aLS)

            if args.anneal == "std":
                kl = log_q_z0 - beta * logp_zk - log_det_jacobians #sum over batches
                loss = beta * torch.abs(torch.mean(log_p_xz)) + kl + penalty_LS + penalty_LF
            elif args.anneal == "off":
                kl = torch.mean(log_q_z0 - logp_zk - log_det_jacobians) #sum over batches
                loss = - torch.mean(log_p_xz) + kl + penalty_LS + penalty_LF
            elif args.anneal == "kl":
                kl = torch.mean(log_q_z0 - logp_zk - log_det_jacobians) #sum over batches
                loss = - torch.mean(log_p_xz) + beta * kl + penalty_LS + penalty_LF
            opt.zero_grad()
            loss.backward(retain_graph=True)
            opt.step()
            train_loss[index] = loss.item()
            t_loss = loss.item()
            i += 1

        TLoss.append(train_loss.mean())
        print('Finish.')
    return test_PLS, test_PLF, test_PBD


