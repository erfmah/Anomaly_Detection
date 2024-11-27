import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.data import RedditDataset
from dgl.nn.pytorch import GATConv
import numpy as np
from sklearn.metrics import roc_auc_score
import random
from utils import *


class GATTransformerLinkPredictor(nn.Module):
    def __init__(self, in_channels, gat_hidden_channels, num_heads, transformer_dim,
                 transformer_heads, transformer_layers, dropout=0.1):
        super(GATTransformerLinkPredictor, self).__init__()

        # GAT layers
        self.gat1 = GATConv(in_channels, gat_hidden_channels, num_heads=num_heads, dropout=dropout)
        self.gat2 = GATConv(gat_hidden_channels * num_heads, gat_hidden_channels, num_heads=1, dropout=dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=gat_hidden_channels,
            nhead=transformer_heads,
            dim_feedforward=transformer_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)

        # Final prediction layers
        self.fc1 = nn.Linear(gat_hidden_channels * 2, gat_hidden_channels)
        self.fc2 = nn.Linear(gat_hidden_channels, 1)

        self.dropout = nn.Dropout(dropout)

    def encode(self, g, features):
        # GAT encoding
        h = self.gat1(g, features)
        h = h.view(h.size(0), -1)  # Flatten the head dimension
        h = F.relu(h)
        h = self.dropout(h)
        h = self.gat2(g, h)
        h = h.squeeze(1)  # Remove head dimension

        # Transformer encoding
        h = self.transformer(h.unsqueeze(0)).squeeze(0)

        return h

    def decode(self, h, edges):
        # Get node pairs for edge prediction
        src, dst = edges
        edge_features = torch.cat([h[src], h[dst]], dim=1)

        # MLP for final prediction
        x = F.relu(self.fc1(edge_features))
        x = self.dropout(x)
        x = self.fc2(x)

        return torch.sigmoid(x)

    def forward(self, g, features, edges):
        h = self.encode(g, features)
        return self.decode(h, edges)
