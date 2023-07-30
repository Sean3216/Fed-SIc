import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

import pandas as pd
import numpy as np

from client import Client
from model import MnistModel
from utils import train, test
from collections import OrderedDict
import tqdm
import copy

class FedHSIc():
    def __init__(self, evalnum, clientpar, perf_threshold):
        self.globalmodel = MnistModel()
        self.selected_clients = []
        self.evaluator = []
        self.final_selected_clients = []
        self.rounds = 0
        self.evalnum = evalnum
        self.clientpar = clientpar
        self.perf_threshold = perf_threshold
        self.params = {}

    #### Vanilla FL Function    
    def aggregate(self):
        modelparams = []
        for i in self.params.keys():
            if (i in self.evaluator) or (i in self.final_selected_clients):
                modelparams.append(self.params[i])
        avg_weights = {}
        for name in modelparams[0].keys():
            avg_weights[name] = torch.mean(torch.stack([w[name] for w in modelparams]), dim = 0)
        self.globalmodel.load_state_dict(avg_weights)

    def clientstrain(self, clientconfig):
        clients = clientconfig
        for i in clients.keys():
            test_client = Client(clients[i])
            test_client.model = copy.deepcopy(self.globalmodel)
            test_client.train()
            self.params[i] = test_client.model.state_dict()
    
    def clientseval(self, clientconfig):
        clientscores = {}
        for report in self.params.keys():
            reportclient = Client(clientconfig[report])
            reportclient.model = MnistModel()
            clientscores[report] = {}
            for cand in self.params.keys():
                if report != cand:
                    reportclient.model.load_state_dict(self.params[cand])
                    clientscores[report][cand] = reportclient.test()['f1']
                else:
                    continue
        return clientscores
        
    #### Proposed Idea Functions
    def get_cand_avg(self,clientsscores):
        clients = pd.DataFrame(clientsscores).fillna(0)
        candidatesavg = {i: sum(clients.loc[i])/(len(clients)-1) for i in range(len(clients))}
        return candidatesavg

    def choose_eval_cand(self, clientconfig):
        clientscores = self.clientseval(clientconfig)
        candidatesavg = self.get_cand_avg(clientscores)
        print("this is candidates avg:")
        print(candidatesavg)
        #### Select evaluators and early candidates
        for i in range(self.evalnum):
            tobeeval = max(candidatesavg, key = candidatesavg.get)
            print("This is to be evaluator!: {}".format(tobeeval))
            candidatesavg.pop(tobeeval,None)
            self.evaluator.append(tobeeval)
            
        #### Assign probability to candidates
        candidatesprob = {i: candidatesavg[i]/sum(candidatesavg.values()) for i in candidatesavg.keys()} # Then rest of the keys left becomes the candidates
        #### Select final candidates based on probability
        np.random.seed(42)
        self.selected_clients = list(np.random.choice(list(candidatesprob.keys()), self.clientpar, p = list(candidatesprob.values()),replace = False))
        print("Candidates probabilities (this round round): {}".format(candidatesprob))
        print("Candidates to be selected: {}".format(self.selected_clients))
        #### Server helps coordinate distance calculation
        print("Calculating Distance!!")
        self.final_selected_clients, clientconfig = self.evalf1(clientconfig, self.selected_clients)
        print("Evaluators and Selected Clients for this round's training is:")
        print("Evaluator: {}".format(self.evaluator))
        print("Selected Clients: {}".format(self.final_selected_clients)) 
    
    def evalf1(self, clientconfig, candidates):
        finer_selected_clients = []
        if len(candidates) != 0:
            for j in candidates:
                sumf1 = 0
                for i in self.evaluator:
                    evalclient = Client(clientconfig[i])
                    selected = Client(clientconfig[j])
                    print("evaluating client {} by evaluator {}!".format(j,i))
                    evalclient.model = MnistModel()
                    selected.model = MnistModel()

                    evalclient.model.load_state_dict(self.params[i])
                    selected.model.load_state_dict(self.params[j])
                    sumf1 += evalclient.evaluate_other_model(copy.deepcopy(selected.model), device = "cuda")
                avgf1 = sumf1/(len(self.evaluator))
                print("against evaluators F1-Score: {}".format(avgf1))
                if avgf1 > self.perf_threshold:
                    finer_selected_clients.append(j)
                else:
                    continue
        else:
            finer_selected_clients = []
        return finer_selected_clients, clientconfig

    def initiate_FL(self, clientconfig, serverdata):
        clients = clientconfig   
        print("Round: {}".format(self.rounds))
        #### Obtain weights
        print("Obtaining Weights!!")
        self.clientstrain(clients)

        #### Server helps choose candidate
        print("Obtaining Candidates!!")
        self.choose_eval_cand(clients)

        #### Aggregate model
        print("Aggregating Model!!")
        self.aggregate()

        #### Replace parameters with global model parameters            
        for i in self.params.keys():
            self.params[i] = self.globalmodel.state_dict()
            
        servertest = torch.utils.data.DataLoader(serverdata, batch_size=clients[0]['batch_size'], shuffle=True)
        loss, results = test(self.globalmodel, servertest, device = "cuda")
        print("Round {} metrics:".format(self.rounds))
        print("Server Loss = {}".format(loss))
        print("Server Accuracy = {}".format(results['acc']))
        print("Round {} finished!".format(self.rounds))
        self.evaluator = []
        self.selected_clients = []
        self.rounds += 1
        return clients, results['acc']  
    
class TopF1():
    def __init__(self, clientpar):
        self.globalmodel = MnistModel()
        self.selected_clients = []
        self.rounds = 0
        self.clientpar = clientpar
        self.params = {}

    #### Vanilla FL Function    
    def aggregate(self):
        modelparams = []
        for i in self.params.keys():
            if i in self.selected_clients:
                modelparams.append(self.params[i])
        avg_weights = {}
        for name in modelparams[0].keys():
            avg_weights[name] = torch.mean(torch.stack([w[name] for w in modelparams]), dim = 0)
        self.globalmodel.load_state_dict(avg_weights)

    def clientstrain(self, clientconfig):
        clients = clientconfig
        for i in clients.keys():
            test_client = Client(clients[i])
            test_client.model = copy.deepcopy(self.globalmodel)
            test_client.train()
            self.params[i] = test_client.model.state_dict()
    
    def clientseval(self, clientconfig):
        clientscores = {}
        for report in self.params.keys():
            reportclient = Client(clientconfig[report])
            reportclient.model = MnistModel()
            clientscores[report] = {}
            for cand in self.params.keys():
                if report != cand:
                    reportclient.model.load_state_dict(self.params[cand])
                    clientscores[report][cand] = reportclient.test()['f1']
                else:
                    continue
        return clientscores
        
    #### Proposed Idea Functions
    def get_cand_avg(self,clientsscores):
        clients = pd.DataFrame(clientsscores).fillna(0)
        candidatesavg = {i: sum(clients.loc[i])/(len(clients)-1) for i in range(len(clients))}
        return candidatesavg

    def choose_eval_cand(self, clientconfig):
        clientscores = self.clientseval(clientconfig)
        candidatesavg = self.get_cand_avg(clientscores)
        print("this is candidates avg:")
        print(candidatesavg)
        #### Select clients to join
        for i in range(self.clientpar):
            tobeeval = max(candidatesavg, key = candidatesavg.get)
            print("This is to be selected!: {}".format(tobeeval))
            candidatesavg.pop(tobeeval,None)
            self.selected_clients.append(tobeeval)
            
        print("Selected Clients for this round's training is:")
        print("Selected Clients: {}".format(self.selected_clients)) 
    
    def initiate_FL(self, clientconfig, serverdata):
        clients = clientconfig   
        print("Round: {}".format(self.rounds))
        #### Obtain weights
        print("Obtaining Weights!!")
        self.clientstrain(clients)

        #### Server helps choose candidate
        print("Obtaining Candidates!!")
        self.choose_eval_cand(clients)

        #### Aggregate model
        print("Aggregating Model!!")
        self.aggregate()

        #### Replace parameters with global model parameters            
        for i in self.params.keys():
            self.params[i] = self.globalmodel.state_dict()
            
        servertest = torch.utils.data.DataLoader(serverdata, batch_size=clients[0]['batch_size'], shuffle=True)
        loss, results = test(self.globalmodel, servertest, device = "cuda")
        print("Round {} metrics:".format(self.rounds))
        print("Server Loss = {}".format(loss))
        print("Server Accuracy = {}".format(results['acc']))
        print("Round {} finished!".format(self.rounds))
        self.selected_clients = []
        self.rounds += 1
        return clients, results['acc']    
    
class FedAvg():
    def __init__(self):
        self.globalmodel = MnistModel()
        self.rounds = 0
        self.params = {}

    #### Vanilla FL Function    
    def aggregate(self):
        modelparams = []
        for i in self.params.keys():
            modelparams.append(self.params[i])
        avg_weights = {}
        for name in modelparams[0].keys():
            avg_weights[name] = torch.mean(torch.stack([w[name] for w in modelparams]), dim = 0)
        self.globalmodel.load_state_dict(avg_weights)

    def clientstrain(self, clientconfig):
        clients = clientconfig
        for i in clients.keys():
            test_client = Client(clients[i])
            test_client.model = copy.deepcopy(self.globalmodel)
            test_client.train()
            self.params[i] = test_client.model.state_dict()
    
    def initiate_FL(self, clientconfig, serverdata):
        clients = clientconfig   
        print("Round: {}".format(self.rounds))
        #### Obtain weights
        print("Obtaining Weights!!")
        self.clientstrain(clients)

        #### Aggregate model
        print("Aggregating Model!!")
        self.aggregate()

        #### Replace parameters with global model parameters            
        for i in self.params.keys():
            self.params[i] = self.globalmodel.state_dict()
            
        servertest = torch.utils.data.DataLoader(serverdata, batch_size=clients[0]['batch_size'], shuffle=True)
        loss, results = test(self.globalmodel, servertest, device = "cuda")
        print("Round {} metrics:".format(self.rounds))
        print("Server Loss = {}".format(loss))
        print("Server Accuracy = {}".format(results['acc']))
        print("Round {} finished!".format(self.rounds))
        self.rounds += 1
        return clients, results['acc']   
    
def Centralized(clientdataset, serverdataset, numround):
    train_loader = torch.utils.data.DataLoader(clientdataset, batch_size = 32)
    test_loader = torch.utils.data.DataLoader(serverdataset, batch_size = 32, shuffle=True)
    model = MnistModel()
    results = train(net = model,
                    trainloader= train_loader,
                    epochs = numround,
                    device = 'cuda',
                    valloader= test_loader)
    
    print(f"Train result centralized: {results}")



    
            
            
        
        