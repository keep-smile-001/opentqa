# -----------------------------------------------
# !/usr/bin/env python
# _*_coding:utf-8 _*_
# @Time: 2020/11/1 17:18
# @File : layer.py
# -----------------------------------------------
import importlib
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm
from block import fusions

"""
This code is modified by Linjie Li from  Remi Cadene's repository.
https://github.com/Cadene/vqa.pytorch
"""

class MuTAN_Attention(nn.Module):
    def __init__(self, dim_v, dim_q, dim_out, method="Mutan", mlp_glimpses=0):
        super(MuTAN_Attention, self).__init__()
        self.mlp_glimpses = mlp_glimpses
        self.fusion = getattr(fusions, method)(
                        [dim_q, dim_v], dim_out, mm_dim=1200,
                        dropout_input=0.1)
        if self.mlp_glimpses > 0:
            self.linear0 = FCNet([dim_out, 512], '', 0)
            self.linear1 = FCNet([512, mlp_glimpses], '', 0)

    def forward(self, q, v):
        alpha = self.process_attention(q, v)

        if self.mlp_glimpses > 0:
            alpha = self.linear0(alpha)
            alpha = F.relu(alpha)
            alpha = self.linear1(alpha)

        alpha = F.softmax(alpha, dim=1) #[8,1,2]

        if alpha.size(2) > 1:  # nb_glimpses > 1
            alphas = torch.unbind(alpha, dim=2)
            v_outs = []
            for alpha in alphas:
                alpha = alpha.unsqueeze(2).expand_as(v)
                v_out = alpha*v
                v_out = v_out.sum(1)
                v_outs.append(v_out)
            v_out = torch.cat(v_outs, dim=1)
        else:
            alpha = alpha.expand_as(v)
            v_out = alpha*v
            v_out = v_out.sum(1)
        return v_out

    def process_attention(self, q, v):
        batch_size = q.size(0)
        n_regions = v.size(1)
        q = q[:, None, :].expand(q.size(0), n_regions, q.size(1))
        alpha = self.fusion([
            q.contiguous().view(batch_size*n_regions, -1),
            v.contiguous().view(batch_size*n_regions, -1) #[2304,360]
        ])
        alpha = alpha.view(batch_size, n_regions, -1) #[64,36,360]
        return alpha


class MuTAN(nn.Module):
    def __init__(self, v_relation_dim, num_hid, num_ans_candidates, gamma):
        super(MuTAN, self).__init__()
        self.gamma = gamma
        self.attention = MuTAN_Attention(v_relation_dim, num_hid,
                                         dim_out=360, method="Mutan",
                                         mlp_glimpses=gamma)
        self.fusion = getattr(fusions, "Mutan")(
                        [num_hid, v_relation_dim*2], num_ans_candidates,
                        mm_dim=1200, dropout_input=0.1)

    def forward(self, v_relation, q_emb):
        # b: bounding box features, not used for this fusion method
        att = self.attention(q_emb, v_relation) #[8,4096] [64,36,1024] #加了att之后的v
        # logits = self.fusion([q_emb, att]) #[64, 3129] #ques和v再一次融合
        # return logits, att

        return att

# ------------------------------
# ------ FC Net ------
# ------------------------------
class FCNet(nn.Module):
    """Simple class for non-linear fully connect network
    """
    def __init__(self, dims, act='ReLU', dropout=0, bias=True):
        super(FCNet, self).__init__()

        layers = []
        for i in range(len(dims)-2):
            in_dim = dims[i]
            out_dim = dims[i+1]
            if 0 < dropout:
                layers.append(nn.Dropout(dropout))
            layers.append(weight_norm(nn.Linear(in_dim, out_dim, bias=bias),
                                      dim=None))
            if '' != act and act is not None:
                layers.append(getattr(nn, act)())
        if 0 < dropout:
            layers.append(nn.Dropout(dropout))
        layers.append(weight_norm(nn.Linear(dims[-2], dims[-1], bias=bias),
                                  dim=None))
        if '' != act and act is not None:
            layers.append(getattr(nn, act)())

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)




# ------------------------------
# ----- Weight Normal MLP ------
# ------------------------------

class MLP(nn.Module):
    """
    Simple class for non-linear fully connect network
    """

    def __init__(self, dims, act='ReLU', dropout_r=0.0):
        super(MLP, self).__init__()

        layers = []
        for i in range(len(dims) - 1):
            in_dim = dims[i]
            out_dim = dims[i + 1]
            if dropout_r > 0:
                layers.append(nn.Dropout(dropout_r))
            layers.append(weight_norm(nn.Linear(in_dim, out_dim), dim=None))
            if act != '':
                layers.append(getattr(nn, act)())

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)



# ------------------------
# -----flatten tensor-----
# ------------------------

class FlattenAtt(nn.Module):
    def __init__(self, cfgs):
        super(FlattenAtt, self).__init__()
        self.cfgs = cfgs

        self.mlp = MLP(
            dims=[cfgs.hidden_size, cfgs.flat_mlp_size, cfgs.flat_glimpse],
            act='',
            dropout_r=cfgs.dropout_r
        )

        self.linear_merge = weight_norm(
            nn.Linear(cfgs.hidden_size * cfgs.flat_glimpse, cfgs.flat_out_size),
            dim=None
        )

    def forward(self, x, x_mask):
        att = self.mlp(x)
        att = att.masked_fill(
            x_mask.unsqueeze(2),
            -1e9
        )
        att = F.softmax(att, dim=1)

        att_list = []
        for i in range(self.cfgs.flat_glimpse):
            att_list.append(
                torch.sum(att[:, :, i: i + 1] * x, dim=1)
            )

        x_atted = torch.cat(att_list, dim=1)
        x_atted = self.linear_merge(x_atted)

        return x_atted
