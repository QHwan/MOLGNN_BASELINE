from __future__ import print_function, division

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter
from torch_geometric.nn import GCNConv, GATConv, GINConv

class GCN(nn.Module):
    def __init__(self, args):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(args.num_node_features, args.dim_enc)
        self.conv2 = GCNConv(args.dim_enc, args.dim_enc)
        self.bn1 = nn.BatchNorm1d(args.dim_enc)
        self.bn2 = nn.BatchNorm1d(args.dim_enc)
        self.fc1 = nn.Linear(args.dim_enc, args.dim_mlp)
        self.fc2 = nn.Linear(args.dim_mlp, args.dim_mlp)
        self.fc3 = nn.Linear(args.dim_mlp, 1)

        self.dropout = args.dropout

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)

        x = torch_scatter.scatter_sum(x, batch, dim=0)

        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.fc2(x)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.fc3(x)

        return x



class GAT(nn.Module):
    def __init__(self, args):
        super(GAT, self).__init__()
        self.conv1 = GATConv(args.num_node_features, args.dim_enc, heads=8, concat=False, dropout=0)
        self.conv2 = GATConv(args.dim_enc, args.dim_enc, heads=8, concat=True, dropout=args.dropout)
        self.fc1 = nn.Linear(args.dim_enc * 8, args.dim_mlp)
        self.fc2 = nn.Linear(args.dim_mlp, args.dim_mlp)
        self.fc3 = nn.Linear(args.dim_mlp, 1)

        self.dropout = args.dropout

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)

        x = torch_scatter.scatter_sum(x, batch, dim=0)

        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.fc2(x)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.fc3(x)

        return x


class GIN(nn.Module):
    def __init__(self, args):
        super(GIN, self).__init__()
        nn1 = nn.Sequential(nn.Linear(args.num_node_features, args.dim_enc),
                            nn.ReLU(),
                            nn.Linear(args.dim_enc, args.dim_enc))
        self.conv1 = GINConv(nn1)
        nn2 = nn.Sequential(nn.Linear(args.dim_enc, args.dim_enc),
                            nn.ReLU(),
                            nn.Linear(args.dim_enc, args.dim_enc))
        self.conv2 = GINConv(nn2)
        self.bn1 = nn.BatchNorm1d(args.dim_enc)
        self.bn2 = nn.BatchNorm1d(args.dim_enc)
        self.fc1 = nn.Linear(args.dim_enc, args.dim_mlp)
        self.fc2 = nn.Linear(args.dim_mlp, args.dim_mlp)
        self.fc3 = nn.Linear(args.dim_mlp, 1)

        self.dropout = args.dropout

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)

        x = torch_scatter.scatter_sum(x, batch, dim=0)

        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.fc2(x)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.fc3(x)

        return x