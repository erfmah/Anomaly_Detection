import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.data import RedditDataset
from dgl.nn.pytorch import GATConv
import numpy as np
from sklearn.metrics import roc_auc_score
import random
from  models import *


def prepare_data():
    # Load Reddit dataset
    dataset = RedditDataset()
    g = dataset[0]

    # Add self-loops
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)

    # Get edges
    edges = g.edges()
    num_edges = g.number_of_edges()

    # Create train/test split for edges
    np.random.seed(42)
    train_mask = np.random.rand(num_edges) < 0.85
    train_edges = (edges[0][train_mask], edges[1][train_mask])
    test_edges = (edges[0][~train_mask], edges[1][~train_mask])

    # Create negative edges for testing
    def create_negative_edges(g, num_neg_edges):
        neg_src = []
        neg_dst = []
        num_nodes = g.number_of_nodes()

        while len(neg_src) < num_neg_edges:
            src = random.randint(0, num_nodes - 1)
            dst = random.randint(0, num_nodes - 1)
            if src != dst and not g.has_edges_between(src, dst):
                neg_src.append(src)
                neg_dst.append(dst)

        return torch.tensor(neg_src), torch.tensor(neg_dst)

    # Create negative edges for training and testing
    num_train_neg = len(train_edges[0])
    num_test_neg = len(test_edges[0])
    train_neg_edges = create_negative_edges(g, num_train_neg)
    test_neg_edges = create_negative_edges(g, num_test_neg)

    return g, dataset.features, train_edges, test_edges, train_neg_edges, test_neg_edges


def train_model(model, g, features, train_edges, train_neg_edges, device, num_epochs=100):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.BCELoss()

    print("Starting training...")
    for epoch in range(num_epochs):
        optimizer.zero_grad()

        # Combine positive and negative edges
        src = torch.cat([train_edges[0], train_neg_edges[0]])
        dst = torch.cat([train_edges[1], train_neg_edges[1]])
        edges = (src, dst)

        # Create labels (1 for positive edges, 0 for negative edges)
        labels = torch.zeros(len(src), device=device)
        labels[:len(train_edges[0])] = 1

        # Forward pass
        pred = model(g, features, edges)
        loss = criterion(pred.view(-1), labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}')

    print("Training completed!")
    return model


def test_model(model, g, features, test_edges, test_neg_edges, device):
    print("\nStarting testing...")
    model.eval()

    with torch.no_grad():
        # Test on original test edges
        src = torch.cat([test_edges[0], test_neg_edges[0]])
        dst = torch.cat([test_edges[1], test_neg_edges[1]])
        edges = (src, dst)

        # Create labels
        labels = torch.zeros(len(src), device=device)
        labels[:len(test_edges[0])] = 1

        # Get predictions
        pred = model(g, features, edges)
        pred = pred.view(-1)

        # Calculate metrics
        predictions = (pred > 0.5).float()
        accuracy = (predictions == labels).float().mean().item()
        auc = roc_auc_score(labels.cpu().numpy(), pred.cpu().numpy())

        print("\nResults on original test set:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"AUC-ROC: {auc:.4f}")

        # Test link addition
        print("\nTesting link addition...")
        new_edges = create_random_edges(g, 1000, add=True)
        test_link_modification(model, g, features, new_edges, True, device)

        # Test link removal
        print("\nTesting link removal...")
        removed_edges = create_random_edges(g, 1000, add=False)
        test_link_modification(model, g, features, removed_edges, False, device)


def create_random_edges(g, num_edges, add=True):
    num_nodes = g.number_of_nodes()
    src_list = []
    dst_list = []

    if add:
        # Create new edges that don't exist
        while len(src_list) < num_edges:
            src = random.randint(0, num_nodes - 1)
            dst = random.randint(0, num_nodes - 1)
            if src != dst and not g.has_edges_between(src, dst):
                src_list.append(src)
                dst_list.append(dst)
    else:
        # Sample existing edges
        existing_edges = g.edges()
        indices = np.random.choice(len(existing_edges[0]), num_edges, replace=False)
        src_list = existing_edges[0][indices].tolist()
        dst_list = existing_edges[1][indices].tolist()

    return torch.tensor(src_list), torch.tensor(dst_list)


def test_link_modification(model, g, features, edges, is_addition, device):
    model.eval()
    with torch.no_grad():
        # Get predictions for modified edges
        pred = model(g, features, edges)
        pred = pred.view(-1)

        # For addition, high predictions indicate good additions
        # For removal, low predictions indicate good removals
        if is_addition:
            quality = pred.mean().item()
            print(f"Average prediction for new links: {quality:.4f}")
        else:
            quality = (1 - pred).mean().item()
            print(f"Average prediction for removed links: {quality:.4f}")
