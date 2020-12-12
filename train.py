from __future__ import print_function, division

import numpy as np
import argparse
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import Data, DataLoader

from sklearn.metrics import mean_squared_error

import batch
import model.model_nn as model

def mse(y, y_pred):
    score = mean_squared_error(y, y_pred)
    return score


def get_batch_list(n_data, n_batch, shuffle):
    data_list = list(range(n_data))
    n_step = int(n_data / n_batch) + 1

    if shuffle:
        np.random.shuffle(data_list)

    return np.array_split(data_list, n_step)




def train(args, loader, model, criterion, optimizer):    
    train_loss_list = []
    model.train()
    for data in loader:
        data = data.to(args.device)

        optimizer.zero_grad()

        y_pred = model(data) 
        loss = criterion(data.y.squeeze(), y_pred.squeeze())
        loss.backward()
        optimizer.step()

        train_loss_list.append(loss.data.item())

    return np.mean(train_loss_list)



def test(args, loader, model, criterion):    
    y_pred_list = []
    y_list = []
    test_loss_list = []
    model.eval()
    for data in loader:
        data = data.to(args.device)

        y_pred = model(data)       
        loss = criterion(data.y.squeeze(), y_pred.squeeze())

        test_loss_list.append(loss.data.item())

        for yp in y_pred.squeeze().cpu().detach().numpy():
            y_pred_list.append(yp)
        for y in data.y.squeeze().cpu().detach().numpy():
            y_list.append(y)

    score = mse(y_list, y_pred_list)

    y_pred_list = np.array(y_pred_list)
    y_list = np.array(y_list)

    return (np.mean(test_loss_list), score, y_list, y_pred_list)
        

parser = argparse.ArgumentParser()
parser.add_argument('--data_file', type=str, default=None)
parser.add_argument('--save_model', type=str, default=None)
parser.add_argument('--save_result', type=str, default=None)
parser.add_argument('--load_model', type=str, default=None)

parser.add_argument('--model', type=str, default=None)
parser.add_argument('--n_layer_graph', type=int, default=2)
parser.add_argument('--n_layer_mlp', type=int, default=3)
parser.add_argument('--dim_enc', type=int, default=256)
parser.add_argument('--dim_mlp', type=int, default=512)

parser.add_argument('--cuda', type=bool, default=True)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--n_epoch', type=int, default=300)
parser.add_argument('--early_stopping', type=int, default=30)
parser.add_argument('--n_batch', type=int, default=48)
parser.add_argument('--fold_idx', type=int, default=0)
args = parser.parse_args()

assert args.data_file is not None
assert args.save_model is not None
assert args.save_result is not None
assert args.model is not None

args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load data
dataset = batch.MoleculeDataset(args)

train_data = dataset.load_data(partition='train')
val_data = dataset.load_data(partition='val')
test_data = dataset.load_data(partition='test')

train_loader = DataLoader(train_data, batch_size=args.n_batch, shuffle=True)
val_loader = DataLoader(val_data, batch_size=args.n_batch, shuffle=True)
test_loader = DataLoader(test_data, batch_size=args.n_batch, shuffle=False)

# Setting experimental environments
if args.model == 'GCN':
    nn_model = model.GCN(args).to(args.device)
elif args.model == 'GAT':
    nn_model = model.GAT(args).to(args.device)
elif args.model == 'GIN':
    nn_model = model.GIN(args).to(args.device)

criterion = nn.MSELoss()
optimizer = optim.Adam(nn_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.95, patience=10, verbose=True)

best_loss = 1e8
patience = 0
if args.load_model is not None:
    checkpoint = torch.load(args.load_model)
    nn_model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    best_loss = checkpoint['best_score']

for i in range(args.n_epoch):
    t_i = time.time()
    train_loss = train(args,
                    loader=train_loader, model=nn_model,
                    criterion=criterion, optimizer=optimizer)
    val_loss, val_score, *_ = test(args, loader=val_loader, model=nn_model, criterion=criterion)
    scheduler.step(val_score)

    print("Time: {:.2f}, Epoch: {}, Train Loss: {:.4f}, Val Loss: {:.4f}, Val Score: {:.4f}, Best Score: {:.4f}".format(time.time() - t_i, i+1, train_loss, val_loss, val_score, best_loss))

    if val_score < best_loss:
        best_loss = val_score
        patience = 0
        torch.save({
            'model_state_dict': nn_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_score': best_loss,
            }, args.save_model)
    else:
        patience += 1

    if patience > args.early_stopping:
        print("Early stopping")
        break

checkpoint = torch.load(args.save_model)
nn_model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

test_loss, test_score, y_list, y_pred_list = test(args, loader=test_loader, model=nn_model, criterion=criterion)
print("Test Score: {:6f}".format(test_score))
np.savez(args.save_result, y=y_list, y_pred=y_pred_list)