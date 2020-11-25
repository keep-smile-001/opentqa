# -----------------------------------------------
# !/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time:2020/9/11 8:32
# @Author:Ma Jie
# -----------------------------------------------

import torch.nn as nn
import torch.nn.functional as F
import torch, math, os
import importlib.util


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
# ---- Net-Utils  --------------
# ------------------------------

class FC(nn.Module):
    def __init__(self, in_size, out_size, dropout_r=0., use_relu=True):
        super(FC, self).__init__()
        self.dropout_r = dropout_r
        self.use_relu = use_relu

        self.linear = nn.Linear(in_size, out_size)

        if use_relu:
            self.relu = nn.ReLU(inplace=True)

        if dropout_r > 0:
            self.dropout = nn.Dropout(dropout_r)

    def forward(self, x):
        x = self.linear(x)

        if self.use_relu:
            x = self.relu(x)

        if self.dropout_r > 0:
            x = self.dropout(x)

        return x


class MLP(nn.Module):
    def __init__(self, in_size, mid_size, out_size, dropout_r=0., use_relu=True):
        super(MLP, self).__init__()

        self.fc = FC(in_size, mid_size, dropout_r=dropout_r, use_relu=use_relu)
        self.linear = nn.Linear(mid_size, out_size)

    def forward(self, x):
        return self.linear(self.fc(x))


class LayerNorm(nn.Module):
    def __init__(self, size, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.eps = eps

        self.a_2 = nn.Parameter(torch.ones(size))
        self.b_2 = nn.Parameter(torch.zeros(size))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class AttFlat(nn.Module):
    def __init__(self, cfgs):
        super(AttFlat, self).__init__()
        self.cfgs = cfgs

        self.mlp = MLP(
            in_size=cfgs.hidden_size,
            mid_size=cfgs.flat_mlp_size,
            out_size=cfgs.flat_glimpses,
            dropout_r=cfgs.dropout_r,
            use_relu=True
        )

        self.linear_merge = nn.Linear(
            cfgs.hidden_size * cfgs.flat_glimpses,
            cfgs.flat_out_size
        )

    def forward(self, x, x_mask):
        att = self.mlp(x)
        att = att.masked_fill(
            x_mask.squeeze(1).squeeze(1).unsqueeze(2),
            -1e9
        )
        att = F.softmax(att, dim=1)

        att_list = []
        for i in range(self.cfgs.flat_glimpses):
            att_list.append(
                torch.sum(att[:, :, i: i + 1] * x, dim=1)
            )

        x_atted = torch.cat(att_list, dim=1)
        x_atted = self.linear_merge(x_atted)

        return x_atted


# ------------------------------
# ---- Multi-Head Attention ----
# ------------------------------

class MHAtt(nn.Module):
    def __init__(self, cfgs):
        super(MHAtt, self).__init__()
        self.cfgs = cfgs

        self.linear_v = nn.Linear(cfgs.hidden_size, cfgs.hidden_size)
        self.linear_k = nn.Linear(cfgs.hidden_size, cfgs.hidden_size)
        self.linear_q = nn.Linear(cfgs.hidden_size, cfgs.hidden_size)
        self.linear_merge = nn.Linear(cfgs.hidden_size, cfgs.hidden_size)

        self.dropout = nn.Dropout(cfgs.dropout_r)

    def forward(self, v, k, q, mask):
        n_batches = q.size(0)

        v = self.linear_v(v).view(
            n_batches,
            -1,
            self.cfgs.multi_head,
            self.cfgs.hidden_size_head
        ).transpose(1, 2)

        k = self.linear_k(k).view(
            n_batches,
            -1,
            self.cfgs.multi_head,
            self.cfgs.hidden_size_head
        ).transpose(1, 2)

        q = self.linear_q(q).view(
            n_batches,
            -1,
            self.cfgs.multi_head,
            self.cfgs.hidden_size_head
        ).transpose(1, 2)

        atted = self.att(v, k, q, mask)
        atted = atted.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            self.cfgs.hidden_size
        )

        atted = self.linear_merge(atted)

        return atted

    def att(self, value, key, query, mask):
        d_k = query.size(-1)

        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)

        return torch.matmul(att_map, value)


# ---------------------------
# ---- Feed Forward Nets ----
# ---------------------------

class FFN(nn.Module):
    def __init__(self, cfgs):
        super(FFN, self).__init__()

        self.mlp = MLP(
            in_size=cfgs.hidden_size,
            mid_size=cfgs.ff_size,
            out_size=cfgs.hidden_size,
            dropout_r=cfgs.dropout_r,
            use_relu=True
        )

    def forward(self, x):
        return self.mlp(x)


# ------------------------
# ---- Self Attention ----
# ------------------------

class SA(nn.Module):
    def __init__(self, cfgs):
        super(SA, self).__init__()

        self.mhatt = MHAtt(cfgs)
        self.ffn = FFN(cfgs)

        self.dropout1 = nn.Dropout(cfgs.dropout_r)
        self.norm1 = LayerNorm(cfgs.hidden_size)

        self.dropout2 = nn.Dropout(cfgs.dropout_r)
        self.norm2 = LayerNorm(cfgs.hidden_size)

    def forward(self, x, x_mask):
        x = self.norm1(x + self.dropout1(
            self.mhatt(x, x, x, x_mask)
        ))

        x = self.norm2(x + self.dropout2(
            self.ffn(x)
        ))

        return x


# -------------------------------
# ---- Self Guided Attention ----
# -------------------------------

class SGA(nn.Module):
    def __init__(self, cfgs):
        super(SGA, self).__init__()

        self.mhatt1 = MHAtt(cfgs)
        self.mhatt2 = MHAtt(cfgs)
        self.ffn = FFN(cfgs)

        self.dropout1 = nn.Dropout(cfgs.dropout_r)
        self.norm1 = LayerNorm(cfgs.hidden_size)

        self.dropout2 = nn.Dropout(cfgs.dropout_r)
        self.norm2 = LayerNorm(cfgs.hidden_size)

        self.dropout3 = nn.Dropout(cfgs.dropout_r)
        self.norm3 = LayerNorm(cfgs.hidden_size)

    def forward(self, x, y, x_mask, y_mask):
        x = self.norm1(x + self.dropout1(
            self.mhatt1(x, x, x, x_mask)
        ))

        x = self.norm2(x + self.dropout2(
            self.mhatt2(y, y, x, y_mask)
        ))

        x = self.norm3(x + self.dropout3(
            self.ffn(x)
        ))

        return x


# ------------------------------------------------
# ---- MAC Layers Cascaded by Encoder-Decoder ----
# ------------------------------------------------

class MCA_ED(nn.Module):
    def __init__(self, cfgs):
        super(MCA_ED, self).__init__()

        self.enc_list = nn.ModuleList([SA(cfgs) for _ in range(cfgs.layer)])
        self.dec_list = nn.ModuleList([SGA(cfgs) for _ in range(cfgs.layer)])

    def forward(self, x, y, x_mask, y_mask):
        # Get hidden vector
        for enc in self.enc_list:
            x = enc(x, x_mask)

        for dec in self.dec_list:
            y = dec(y, x, y_mask, x_mask)

        return x, y
