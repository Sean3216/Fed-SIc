import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import time

from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score

def train(net, trainloader: torch.utils.data.DataLoader, epochs: int,device: torch.device, valloader: torch.utils.data.DataLoader = None) -> None:
    """Train the network."""
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr = 0.01)

    print(f"Training {epochs} epoch(s) w/ {len(trainloader)} batches each")
    start_time = time.time()
    
    net.to(device)
    net.train()
    # Train the network
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        total_loss = 0.0
        for i, (x, y) in enumerate(trainloader):
            data, label = x.to(device), y.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(data)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()

            # print statistics
            #running_loss += loss.item()
            total_loss += loss.item()
            
            #if i % 100 == 99:  # print every 100 mini-batches
            #    print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 100))
            #    running_loss = 0.0
        #validation accuracy
        _ , epoch_results = test(net, valloader, device)
        print(f"Epoch {epoch} Result: {epoch_results}") 
        net.to(device)
        net.train()       
        total_loss = total_loss / len(trainloader)

    total_time = time.time() - start_time
    net.to("cpu")  # move model back to CPU
    
    # metrics
    val_loss = 0.0
    val_acc, val_f1, val_rec, val_prec = 0.0, 0.0, 0.0, 0.0

    train_loss, train_results = test(net, trainloader, device)
    if valloader:
        val_loss, test_results = test(net, valloader, device)
        val_acc = test_results["acc"]
        val_f1 = test_results["f1"]
        val_rec = test_results["rec"]
        val_prec = test_results["prec"]

    results = {
        "training_time": total_time,
        "train_loss": train_loss,
        "train_acc": train_results["acc"],
        "train_rec":train_results["rec"],
        "train_f1":train_results["f1"],
        "train_prec":train_results["prec"],
        "validation_loss": val_loss,
        "validation_acc": val_acc,
        "validation_f1":val_f1,
        "validation_rec":val_rec,
        "validation_prec": val_prec,
    }

    #print(f"Epoch took: {total_time:.2f} seconds")
    return results


def test(net, testloader,device: str = "cpu"):
    """Validate the network on the entire test set."""
    criterion = nn.CrossEntropyLoss()
    loss = 0.0
    
    net.to(device)
    net.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(testloader):
            data, labels = x.to(device), y.to(device)
            outputs = net(data)
            loss += criterion(outputs, labels)
            _, predicted = torch.max(outputs, 1)
            # appending
            y_pred.extend(predicted.cpu().detach().numpy())
            y_true.extend(labels.cpu().detach().numpy())
    
    loss = loss / len(testloader)
    net.to("cpu")  # move model back to CPU
    
    # convert tensors to numpy arrays
    y_true = np.array(y_true,dtype=np.int64)
    y_pred = np.array(y_pred,dtype=np.int64)

    # calculate accuracy
    acc = accuracy_score(y_true, y_pred)
    # calculate precision
    precision = precision_score(y_true, y_pred, average='micro')
    # calculate recall
    recall = recall_score(y_true, y_pred, average='micro')
    # calculate F1-score
    f1 = f1_score(y_true, y_pred, average='micro')

    results = {
        "acc":acc,
        "prec":precision,
        "rec":recall,
        "f1":f1,
    }
    
    return loss, results