from client import Client
from model import MnistModel
from datasets import create_datasets
from Server import FedHSIc, TopF1, FedAvg, Centralized 

from utils import test

import pandas as pd
import numpy as np
import random
import argparse

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from collections import OrderedDict
import copy

parser = argparse.ArgumentParser(description = 'Choosing which experiment scenario to do')
group = parser.add_mutually_exclusive_group()
group.add_argument('--fedavg', help = 'original FedAvg', action= argparse.BooleanOptionalAction)
group.add_argument('--topf1', help = 'select client based on highest f1', action= argparse.BooleanOptionalAction)
group.add_argument('--centralized', help = 'perform a centralized version of the training', action= argparse.BooleanOptionalAction)
group.add_argument('--fedhsic', help = 'proposed method', action= argparse.BooleanOptionalAction)

parser.add_argument('--numrounds', default = 10, help = 'number of rounds', type = int)
parser.add_argument('--numclients', help = 'total available clients', type = int)
parser.add_argument('--candidateclientnum', help = 'number of clients to pick', type = int)
parser.add_argument('--nummalicious', help = 'the number of malicious clients among all clients', type = int)
parser.add_argument('--numeval', help = 'number of evaluators to choose from existing clients', type = int)
parser.add_argument('--f1_threshold', help = 'the performance threshold to help select the clients', type = float)

opt = parser.parse_args()

def main(numrounds: int = 10, numclients: int = 10, candidateclientnum: int = 5 , 
         numeval: int = 2, perf_threshold: float = 0.5,
         nummalicious: int = 0, noisemalicious: float = 40, iid: bool = True):  
    
    #### Initiate clients and data
    client_datasets, server_datasets = create_datasets(
            data_path='./data', 
            dataset_name='MNIST', 
            num_clients=numclients, 
            num_shards=200, 
            iid=iid, 
            print_count = True,
            num_malicious_clients= nummalicious,
            noise_level=noisemalicious 
            )
    clients = {}
    random.seed(42)
    for i in range(numclients):
        clients[i] = {"id": i,
                      "train_data": client_datasets[i],
                      "val_size": random.uniform(0,0.3),
                      "batch_size": 32,
                      "local_epoch": random.randint(3,5)}
            
    if opt.fedhsic:
        #### initiate federated learning rounds
        server = FedHSIc(numeval, candidateclientnum, perf_threshold) ### initialize server
        allaccuracy = []
        for i in range(numrounds):
            clients, acc = server.initiate_FL(clients, server_datasets)
            allaccuracy.append(acc)
        print("Accuracy of all rounds: {}".format(allaccuracy))

    if opt.fedavg:
        #### initiate federated learning rounds
        server = FedAvg() ### initialize server
        allaccuracy = []
        for i in range(numrounds):
            clients, acc = server.initiate_FL(clients, server_datasets)
            allaccuracy.append(acc)
        print("Accuracy of all rounds: {}".format(allaccuracy))

    if opt.topf1:
        #### initiate federated learning rounds
        server = TopF1(candidateclientnum) ### initialize server
        allaccuracy = []
        for i in range(numrounds):
            clients, acc = server.initiate_FL(clients, server_datasets)
            allaccuracy.append(acc)
        print("Accuracy of all rounds: {}".format(allaccuracy))
         

if __name__ == "__main__":
    if opt.centralized:
         #### Initiate clients and data
         client_datasets, server_datasets = create_datasets(
            data_path='./data', 
            dataset_name='MNIST', 
            num_clients=1, 
            num_shards=200, 
            iid=True, 
            print_count = True,
            num_malicious_clients=0
            )
         
         #### initiate centralized learning
         Centralized(client_datasets[0], server_datasets, opt.numrounds)
    else:
        main(numclients= opt.numclients, candidateclientnum= opt.candidateclientnum, perf_threshold= opt.f1_threshold,
             nummalicious= opt.nummalicious, numeval= opt.numeval, numrounds = opt.numrounds)
