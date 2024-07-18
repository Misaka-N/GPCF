import torch
import torch.nn as nn
import torch.nn.init as init
from models.encoder import neighbor_feats_sampler

class PromptComponent(nn.Module):

    def __init__(self, prompt_num, prompt_dim, input_dim):
        super(PromptComponent, self).__init__()

        # Initialize
        self.prompt_num = prompt_num
        self.prompt_dim = prompt_dim
        self.input_dim = input_dim

        self.prompt = nn.Parameter(torch.Tensor(self.prompt_num, self.prompt_dim))
        init.orthogonal_(self.prompt)
    
    def feat_prompt(self, feat):
        sim_matrix = torch.matmul(feat, self.prompt.t())
        feat_emb = feat * torch.matmul(sim_matrix, self.prompt)

        return feat_emb

    def forward(self, g, block):
        center_prompt = self.feat_prompt(block.dstdata['feat'])
        block.srcdata['feat'] = self.feat_prompt(block.srcdata['feat'])
        neighbor_prompt = neighbor_feats_sampler(g, block)

        prompt_emb = torch.concat((center_prompt, neighbor_prompt), dim=-1)

        return prompt_emb


class AnswerFunc(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(AnswerFunc, self).__init__()

        self.layers = 2
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.func = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.PReLU(),
            nn.Linear(hidden_dim, output_dim))
        
        for _, layer in enumerate(self.func):
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=0.0, std=1.0)
                nn.init.normal_(layer.bias, mean=0.0, std=1.0)

    def forward(self, feat):
        logits = self.func(feat)

        return logits