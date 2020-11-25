#-----------------------------------------------
#!/usr/bin/env python
# _*_ coding: utf-8 _*_
#@Time:2020/10/19 9:58
#@Author:Ma Jie
#-----------------------------------------------

import os, torch
import importlib.util
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm

class SimCLR(nn.Module):
    '''
    use simclr to obtain diagram representations.
    '''

    def __init__(self, cfgs):
        super(SimCLR, self).__init__()
        self.cfgs = cfgs

        spec = importlib.util.spec_from_file_location(
            "simclr",
            os.path.join(self.cfgs.simclr['checkpoints_folder'], 'resnet_simclr.py'))

        resnet_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(resnet_module)
        self.simclr_resnet = resnet_module.ResNetSimCLR(self.cfgs.simclr['base_model'],
                                                        int(self.cfgs.simclr['out_dim']))
        # map_location must include 'cuda:0', otherwise it will occur error
        state_dict = torch.load(os.path.join(self.cfgs.simclr['checkpoints_folder'], 'model.pth'),
                                map_location='cuda:{}'.format(self.cfgs.gpu) if torch.cuda.is_available() else 'cpu')
        self.simclr_resnet.load_state_dict(state_dict)

        if self.cfgs.not_fine_tuned in 'True':
            for param in self.simclr_resnet.parameters():
                param.requires_grad = False

    def forward(self, dia):
        dia_feats, _ = self.simclr_resnet(dia)
        return dia_feats

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
            nn.Linear(cfgs.hidden_size * cfgs.flat_glimpse, cfgs.hidden_size),
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
        self.v_net = MLP([cfgs.dia_feat_size,
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
        # batch_size = v.shape[0]
        # feat_dim = v.shape[-1]
        # v = v.reshape(batch_size, -1, feat_dim)
        v = v.unsqueeze(1)

        att, logits = self.BiAtt(v, q)  # b x g x v x q

        for g in range(self.cfgs.glimpse):
            bi_emb = self.b_net[g].forward_with_weights(
                v, q, att[:, g, :, :])  # b x l x h
            q = self.q_prj[g](bi_emb.unsqueeze(1)) + q

        return q