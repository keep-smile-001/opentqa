# -----------------------------------------------
# !/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time:2020/9/11 8:32
# -----------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm

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


# ------------------------------
# ------ Bilinear Connect ------
# ------------------------------

class BC(nn.Module):
    """
    Simple class for non-linear bilinear connect network
    """

    def __init__(self, cfgs, atten=False):
        super(BC, self).__init__()

        self.cfgs = cfgs
        self.v_net = MLP([cfgs.cp_feat_size,
                          cfgs.ba_hidden_size], dropout_r=cfgs.dropout_r)
        self.q_net = MLP([cfgs.hidden_size,
                          cfgs.ba_hidden_size], dropout_r=cfgs.dropout_r)
        if not atten:
            self.p_net = nn.AvgPool1d(cfgs.k_times, stride=cfgs.k_times)
        else:
            self.dropout = nn.Dropout(cfgs.classifer_dropout_r)  # attention

            self.h_mat = nn.Parameter(torch.Tensor(1, cfgs.glimpse, 1, cfgs.ba_hidden_size).normal_())
            self.h_bias = nn.Parameter(torch.Tensor(1, cfgs.glimpse, 1, 1).normal_())

    def forward(self, v, q):
        batch_size = v.shape[0]
        # low-rank bilinear pooling using einsum
        v_ = self.dropout(self.v_net(v))
        q_ = self.q_net(q)

        logits = torch.einsum('xhyk,bvk,bqk->bhvq',
                              [self.h_mat, v_, q_]) + self.h_bias
        return logits  # b x h_out x v x q (hadamard product, matrix product)

    def forward_with_weights(self, v, q, w):
        v_ = self.v_net(v)  # b x v x d
        q_ = self.q_net(q)  # b x q x d

        logits = torch.einsum('bvk,bvq,bqk->bk', [v_, w, q_])
        logits = logits.unsqueeze(1)  # b x 1 x d
        logits = self.p_net(logits).squeeze(1) * self.cfgs.k_times  # sum-pooling
        return logits


# ------------------------------
# -------- BiAttention ---------
# ------------------------------


class BiAttention(nn.Module):
    def __init__(self, cfgs):
        super(BiAttention, self).__init__()

        self.cfgs = cfgs
        self.logits = weight_norm(BC(cfgs, True), name='h_mat', dim=None)

    def forward(self, v, q, v_mask=True, logit=False, mask_with=-float('inf')):
        v_num = v.size(1)
        q_num = q.size(1)
        logits = self.logits(v, q)  # b x g x v x q

        if v_mask:
            mask = (0 == v.abs().sum(2)).unsqueeze(
                1).unsqueeze(3).expand(logits.size())
            logits.data.masked_fill_(mask.data, mask_with)

        if not logit:
            p = nn.functional.softmax(
                logits.view(-1, self.cfgs.glimpse, v_num * q_num), 2)
            return p.view(-1, self.cfgs.glimpse, v_num, q_num), logits

        return logits


# ------------------------------
# - Bilinear Attention Network -
# ------------------------------

class BAN(nn.Module):
    def __init__(self, cfgs):
        super(BAN, self).__init__()

        self.cfgs = cfgs
        self.BiAtt = BiAttention(cfgs)
        self.b_net = BC(cfgs)
        self.q_prj = MLP([cfgs.hidden_size, cfgs.hidden_size], '', cfgs.dropout_r)
        b_net = []
        q_prj = []
        for i in range(cfgs.glimpse):
            b_net.append(BC(cfgs))
            q_prj.append(MLP([cfgs.hidden_size, cfgs.hidden_size], '', cfgs.dropout_r))
        self.b_net = nn.ModuleList(b_net)
        self.q_prj = nn.ModuleList(q_prj)

    def forward(self, q, v):
        att, logits = self.BiAtt(v, q)  # b x g x v x q

        for g in range(self.cfgs.glimpse):
            bi_emb = self.b_net[g].forward_with_weights(
                v, q, att[:, g, :, :])  # b x l x h
            q = self.q_prj[g](bi_emb.unsqueeze(1)) + q

        return q