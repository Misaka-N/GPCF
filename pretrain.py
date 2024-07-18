import torch
import dgl
from dgl.dataloading import MultiLayerNeighborSampler
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from random import shuffle
from utils.args import get_pretrain_args
from utils.dataloader import pretrain_dataloader
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, add_self_loops
from torch_geometric.loader.cluster import ClusterData
from utils.tools import EarlyStopping, set_random
from models.encoder import GCNEncoder, MoCo
import torch.nn.functional as F
import warnings
import copy
import gc

def meta_train(g, support_task_loader, model):
    task_loss, task_nodes = 0., 0
    for step, (input_nodes, seeds, blocks) in enumerate(support_task_loader):
        blocks = [block.to(device) for block in blocks]
        loss, proj_h, moco_h = model(g, blocks)
        task_loss += loss
        task_nodes += len(blocks[-1].dstdata)
    task_loss = task_loss / task_nodes

    return task_loss, proj_h, moco_h 


def nodes_sampler(g, sample_shots, num_classes):
    num_nodes = g.num_nodes()
    sample_num_per_class = sample_shots

    degs = g.in_degrees().float()

    labels = torch.zeros(num_nodes, dtype=torch.long)
    for i in range(num_classes):
        nodes_in_class = torch.where(g.ndata['label'] == i)[0]
        labels[nodes_in_class] = i

    sorted_nodes = {i: torch.tensor([], dtype=torch.long) for i in range(num_classes)}
    for i in range(num_classes):
        nodes_in_class = torch.where(labels == i)[0]
        sorted_nodes[i] = nodes_in_class[degs[nodes_in_class].argsort(descending=True)]
        if len(sorted_nodes[i]) < sample_num_per_class:
            warnings.warn(f'the node num of class{i} ({len(sorted_nodes[i])}) is less than {sample_num_per_class}')

    sampled_nodes_idx = [sorted_nodes[i][:min(len(sorted_nodes[i]), sample_num_per_class)] for i in range(num_classes)]
    sampled_nodes_idx = torch.cat(sampled_nodes_idx)

    return sampled_nodes_idx


if __name__ == "__main__":
    args = get_pretrain_args()

    if torch.cuda.is_available():
        print("CUDA is available")
        print("CUDA version:", torch.version.cuda)
        device = torch.device("cuda")
        set_random(args.seed, True)
    else:
        print("CUDA is not available")
        device = torch.device("cpu")
        set_random(args.seed, False)

    # Get pre-train datasets
    neighbor_sampler = MultiLayerNeighborSampler([args.neighbor_num] * args.neighbor_layer)

    node_loader = [] 
    for dataset in args.dataset:
        print("---Downloading dataset: " + dataset + "---")
        g, dataname, num_classes = pretrain_dataloader(input_dim=args.input_dim, dataset=dataset)
        sampled_nodes_idx = nodes_sampler(g, args.sample_shots, num_classes)
        neighbor_dataloader = dgl.dataloading.DataLoader(g, sampled_nodes_idx, neighbor_sampler, device=device, use_ddp=False, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=0)
        node_loader.append(neighbor_dataloader)

    # Pre-train setup
    print("---Pretrain GNN---")
    model = GCNEncoder(input_dim=args.input_dim,
                       hidden_dim=args.hidden_dim,
                       output_dim=args.output_dim,
                       gnn_layer=args.gnn_layer,
                       projector_layer=args.projector_layer,
                       lr=args.lr,
                       temper=args.temperature,
                       weight=args.weight,
                       ).to(device)
    moco_updater = MoCo(bias=args.moco_bias)
    optimizer = optim.Adam(list(model.parameters()), lr=args.lr, weight_decay=args.decay)
    early_stopper = EarlyStopping(path=args.path, patience=args.patience, min_delta=0)

    # Meta-learning for pre-train
    for epoch in range(args.max_epoches):
        meta_loss = 0.

        for task_id, support_task_loader in enumerate(node_loader):
            meta_model = copy.deepcopy(model)

            for step in range(args.adapt_step):
                adapt_loss, proj_h, moco_h = meta_train(g, support_task_loader, meta_model)
                print("Epoch: {} | Task: {} | Step: {} | Adapt-Loss: {:.5f}".format(epoch, task_id, step, adapt_loss.item()))
                meta_model.adapt(adapt_loss)

            update_loss, proj_h, moco_h = meta_train(g, support_task_loader, model)
            print("Epoch: {} | Task: {} | Update-Loss: {:.5f}".format(epoch, task_id, update_loss.item()))
            meta_loss += update_loss
        
        meta_loss = meta_loss / len(node_loader)
        print("Epoch: {} | Meta-Loss: {:.5f}".format(epoch, meta_loss.item()))

        early_stopper(model.conv_layers.state_dict(), meta_loss)
        if early_stopper.early_stop:
            print("Stopping training...")
            print("Best Score: ", early_stopper.best_score)
            break

        optimizer.zero_grad()
        meta_loss.backward()
        optimizer.step()

        moco_updater.momentum_update(model.moco_conv_layers, model.conv_layers)
