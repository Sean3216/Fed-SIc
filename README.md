# Fed-SIc
Repo contains the codes used for experiments done in Fed-HSIc: Horizontally Selecting Important Clients for Federated Learning

To run the experiments, navigate to project directory and execute this from command prompt:
### (Extreme Condition) ###
```
python Main.py --fedavg --numrounds 10 --numclients 10 -- candidateclientnum 5 --nummalicious 7 --numeval 2 --f1_threshold 0.9
```

Multiple strategies available are:
* Fed-Avg
* TopF1 (called as F1S-FL in the paper)
* Centralized
* Fed-HSIc (or known as Fed-SIc)

Scenarios for experiment can be controlled by modifying the settings of the experiment like numrounds, numclients, candidateclientnum, nummalicious, numeval, and f1_threshold
