import numpy as np
import torch
import torch.nn.functional as F

from torch.utils.data.sampler import SubsetRandomSampler
from utils import train, test

class Client():
    def __init__(self, client_config:dict):
        # client config as dict to make configuration dynamic
        self.id = client_config["id"]
        self.config = client_config
        self.__model = None
        
        # check if CUDA is available
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        # if we use validation
        if self.config["val_size"] > 0.0:
            num_train = len(self.config["train_data"])
            indices = list(range(num_train))
            np.random.seed(42)
            np.random.shuffle(indices)
            split = int(np.floor(self.config["val_size"] * num_train))
            train_idx, valid_idx = indices[split:], indices[:split]

            # define samplers for obtaining training and validation batches
            torch.manual_seed(42)
            train_sampler = SubsetRandomSampler(train_idx)
            valid_sampler = SubsetRandomSampler(valid_idx)
            
            # prepare data loaders (combine dataset and sampler)
            self.train_loader = torch.utils.data.DataLoader(self.config["train_data"], 
                                                            batch_size=self.config["batch_size"],
                                                            sampler=train_sampler)
            self.valid_loader = torch.utils.data.DataLoader(self.config["train_data"],
                                                            batch_size=self.config["batch_size"],
                                                            sampler=valid_sampler) 
        else:
            self.train_loader = torch.utils.data.DataLoader(self.config["train_data"], 
                                                            batch_size=self.config["batch_size"])
            self.valid_loader = None

    @property
    def model(self):
        return self.__model

    @model.setter
    def model(self, model):
        self.__model = model

    def __len__(self):
        """Return a total size of the client's local data."""
        return len(self.train_loader.sampler)
    
    def train(self):
        results = train(net=self.model, 
                        trainloader= self.train_loader, 
                        epochs= self.config["local_epoch"],
                        device= self.device, 
                        valloader= self.valid_loader)
        print(f"Train result client {self.id}: {results}")
    
    def test(self):
        loss,result = test(net = self.model, 
                        testloader = self.valid_loader,
                        device=self.device)
        #print(f"Test result client {self.id}: {loss, result}")
        return result

    def evaluate_other_model (self, other_model, device='cpu'):
        # Use evaluator local data, one model is the evaluator's, other is the client being evaluated
         
        #copy of evaluator model
        loss, result = test(net=other_model,
                            testloader=self.train_loader,
                            device=device)
        del other_model #delete other model
        return result['f1']
