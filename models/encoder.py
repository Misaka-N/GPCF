import torch
import dgl
import torch.nn as nn
from dgl.nn import  GraphConv
import random
import copy
import traceback
from torch.nn.functional import normalize
import numpy as np
import torch.nn.functional as F
from utils.tools import clone_module, update_module
from torch.autograd import grad
from torch.nn.functional import normalize, cosine_similarity

def set_grad(module, val):
    for submodule in module:
        for p in submodule.parameters():
            p.requires_grad = val


def neighbor_feats_sampler(g, block):

    neighbor_node_features_avg_list = []
    for central_node_index in range(len(block.dstdata['feat'])):
        input_edges = block.in_edges(central_node_index, form='eid')
        src_local_ids, _ = block.find_edges(input_edges)
        neighbor_node_features = block.srcdata['feat'][src_local_ids]
        neighbor_node_features_avg = torch.mean(neighbor_node_features, dim=0)
        neighbor_node_features_avg_nor = normalize(neighbor_node_features_avg, p=2, dim=-1)
        neighbor_node_features_avg_list.append(neighbor_node_features_avg_nor)

    neighbor_node_features_avg_matrix = torch.stack(neighbor_node_features_avg_list)

    return neighbor_node_features_avg_matrix


class Projector(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(Projector, self).__init__()

        self.num_layers = num_layers
        
        self.linears = torch.nn.ModuleList()
        self.linears.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.linears.append(nn.Linear(hidden_dim, hidden_dim))
        self.linears.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, x):

        x_emb = self.linears[0](x)

        for i in range(self.num_layers - 1):
            x_emb = F.relu(self.linears[i + 1](x_emb))
        
        x_emb = self.linears[self.num_layers - 1](x_emb)

        return x_emb


class MoCo():
    def __init__(self, bias):
        super().__init__()
        self.bias = bias

    def momentum_update(self, model_old, model_new):
        for params_old, params_new in zip(model_old.parameters(), model_new.parameters()):
            if params_old.data == None:
                params_old.data = params_new.data
            else:
                params_old.data = self.bias * params_old.data + (1 - self.bias) * params_new.data


class GCNEncoder(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, gnn_layer, projector_layer, lr, temper, weight, first_order=False, allow_unused=None, allow_nograd=False):
        super(GCNEncoder, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.gnn_layer = gnn_layer
        self.projector_layer = projector_layer
        self.lr = lr
        self.temper = temper
        self.weight = weight

        self.first_order = first_order
        self.allow_nograd = allow_nograd
        if allow_unused is None:
            allow_unused = allow_nograd
        self.allow_unused = allow_unused

        # GCN Encoder (fast update)
        self.conv_layers = torch.nn.ModuleList([GraphConv(self.input_dim , self.hidden_dim)])
        for _ in range(self.gnn_layer - 2):
            self.conv_layers.append(GraphConv(self.hidden_dim , self.hidden_dim))
        self.conv_layers.append(GraphConv(self.hidden_dim , self.output_dim))

        # MoCo GCN Encoder (low update)
        self.moco_conv_layers = copy.deepcopy(self.conv_layers)
        set_grad(self.moco_conv_layers, False) # disable grad backward

        # Projector
        self.projector = Projector(self.output_dim, self.output_dim, self.output_dim, self.projector_layer)


    def meta_update(self, module, lr, grads=None):
        if grads is not None:
            params = list(module.parameters())
            if not len(grads) == len(list(params)):
                msg = 'WARNING:maml_update(): Parameters and gradients have different length. ('
                msg += str(len(params)) + ' vs ' + str(len(grads)) + ')'
                print(msg)
            for p, g in zip(params, grads):
                if g is not None:
                    p.update = - lr * g
        return update_module(module)


    def adapt(self, loss, first_order=None, allow_unused=None, allow_nograd=None):
        if first_order is None:
            first_order = self.first_order
        if allow_unused is None:
            allow_unused = self.allow_unused
        if allow_nograd is None:
            allow_nograd = self.allow_nograd
        second_order = not first_order

        parameters = list(self.conv_layers.parameters()) + list(self.projector.parameters())

        gnn_param_count = sum(1 for _ in self.conv_layers.parameters())

        if allow_nograd:
            diff_params = [p for p in parameters if p.requires_grad]
            grad_params = grad(loss, diff_params, retain_graph=second_order, create_graph=second_order, allow_unused=allow_unused)
            gradients = []
            grad_counter = 0

            for param in parameters:
                if param.requires_grad:
                    gradient = grad_params[grad_counter]
                    grad_counter += 1
                else:
                    gradient = None
                gradients.append(gradient)
        else:
            try:
                gradients = grad(loss, parameters, retain_graph=second_order, create_graph=second_order, allow_unused=allow_unused)
            except RuntimeError:
                traceback.print_exc()
                print('Maybe try with allow_nograd=True and/or allow_unused=True ?')

        gnn_gradients = gradients[:gnn_param_count]
        projector_gradients = gradients[gnn_param_count:]

        self.conv_layers = self.meta_update(self.conv_layers, self.lr, gnn_gradients)
        self.projector = self.meta_update(self.projector, self.lr, projector_gradients)   

    def contrastive_loss(self, g, blocks, proj_h, moco_h):
        neighbor_feats = neighbor_feats_sampler(g, blocks[-1])

        # Center loss
        center_sim = torch.mm(proj_h, moco_h.t()) / self.temper
        center_sim_exp = torch.exp(center_sim)
        center_loss = -torch.log(torch.diag(center_sim_exp) / torch.sum(center_sim_exp, dim=-1))
        center_loss_all = torch.sum(center_loss, dim=-1)

        # Neighbor loss
        neighbor_sim = torch.mm(proj_h, neighbor_feats.t()) / self.temper
        neighbor_sim_exp = torch.exp(neighbor_sim)
        neighbor_loss = -torch.log(torch.diag(neighbor_sim_exp) / torch.sum(neighbor_sim_exp, dim=-1))
        neighbor_loss_all = torch.sum(neighbor_loss, dim=-1)

        loss = center_loss_all + self.weight * neighbor_loss_all

        return loss
    
    def aggregation(self, g, blocks):
        # GNN aggregation & projection
        # for i, block in enumerate(blocks):
        #     conv_h = self.conv_layers[i](block, conv_h)

        # for i, block in enumerate(blocks):
        #     moco_h = self.conv_layers[i](block, moco_h)
        for i in range(len(blocks)):
            blocks[i].dstdata['feat'] = self.conv_layers[i](blocks[i], blocks[i].srcdata['feat'])
            if i != self.gnn_layer - 1:
                blocks[i + 1].srcdata['feat'] = blocks[i].dstdata['feat'] 

        conv_h = blocks[-1].dstdata['feat']

        for i in range(len(blocks)):
            blocks[i].dstdata['feat'] = self.moco_conv_layers[i](blocks[i], blocks[i].srcdata['feat'])
            if i != self.gnn_layer - 1:
                blocks[i + 1].srcdata['feat'] = blocks[i].dstdata['feat'] 

        moco_h = blocks[-1].dstdata['feat']

        conv_h = normalize(conv_h, p=2, dim=-1)
        moco_h = normalize(moco_h, p=2, dim=-1)
        proj_h = normalize(self.projector(conv_h), p=2, dim=-1)

        return proj_h, moco_h

    def forward(self, g, blocks):
        proj_h, moco_h = self.aggregation(g, blocks)
        loss = self.contrastive_loss(g, blocks, proj_h, moco_h)

        return loss, proj_h, moco_h


class GCN(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, gnn_layer):
        super(GCNEncoder, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.gnn_layer = gnn_layer

        # GCN
        self.conv_layers = torch.nn.ModuleList([GraphConv(self.input_dim , self.hidden_dim)])
        for _ in range(self.gnn_layer - 2):
            self.conv_layers.append(GraphConv(self.hidden_dim , self.hidden_dim))
        self.conv_layers.append(GraphConv(self.hidden_dim , self.output_dim))
        

    def forward(self, g, blocks):
        for i in range(len(blocks)):
            blocks[i].dstdata['feat'] = self.conv_layers[i](blocks[i], blocks[i].srcdata['feat'])
            if i != self.gnn_layer - 1:
                blocks[i + 1].srcdata['feat'] = blocks[i].dstdata['feat'] 

        return blocks[-1]
