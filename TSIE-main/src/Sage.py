import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.nn as dglnn
import numpy as np


class SAGE(nn.Module):
    def __init__(self, in_feats, n_hidden, aggr, dropout_p):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, aggr))
        # self.layers.append(dglnn.SAGEConv(n_hidden*2, n_hidden, aggr))
        self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, aggr))
        self.dropout = nn.Dropout(dropout_p)
        self.n_hidden = n_hidden

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            h = h.view(h.shape[0], -1)
            if l != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h