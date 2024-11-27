import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Reddit
from torch_geometric.utils import train_test_split_edges, to_dense_adj
from torch_geometric.nn import GATConv
import numpy as np
from torch.utils.data import DataLoader
from utils import *
from models import *


def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Prepare data
    print("Loading Reddit dataset...")
    g, features, train_edges, test_edges, train_neg_edges, test_neg_edges = prepare_data()
    g = g.to(device)
    features = features.to(device)

    # Move edges to device
    train_edges = (train_edges[0].to(device), train_edges[1].to(device))
    test_edges = (test_edges[0].to(device), test_edges[1].to(device))
    train_neg_edges = (train_neg_edges[0].to(device), train_neg_edges[1].to(device))
    test_neg_edges = (test_neg_edges[0].to(device), test_neg_edges[1].to(device))

    # Initialize model
    print("Initializing model...")
    model = GATTransformerLinkPredictor(
        in_channels=features.shape[1],
        gat_hidden_channels=64,
        num_heads=4,
        transformer_dim=256,
        transformer_heads=8,
        transformer_layers=2,
        dropout=0.1
    ).to(device)

    # Train model
    model = train_model(model, g, features, train_edges, train_neg_edges, device)

    # Test model
    test_model(model, g, features, test_edges, test_neg_edges, device)

    return model


if __name__ == "__main__":
    model = main()