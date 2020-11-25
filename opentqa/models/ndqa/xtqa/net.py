# -----------------------------------------------
# !/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time:2020/10/19 9:59
# @Author:Ma Jie
# -----------------------------------------------

import torch
import torch.nn as nn
from opentqa.models.dqa.xtqa.layer import FlattenAtt, MLP, BAN, weight_norm
from utils.tools import make_mask, get_span_feat, get_ix


class Net(nn.Module):
    def __init__(self, cfgs, pretrained_emb, token_size):
        super(Net, self).__init__()
        self.cfgs = cfgs

        self.embedding = nn.Embedding(
            token_size,
            cfgs.word_emb_size
        )

        if cfgs.pretrained_emb['name']:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_emb))

        # use gru to encode languages
        self.encode_lang = nn.GRU(
            input_size=cfgs.word_emb_size,
            hidden_size=cfgs.hidden_size,
            batch_first=True,
        )

        # flatten tensor
        self.flat = FlattenAtt(cfgs)

        self.sigmoid = nn.Sigmoid()

        self.pool2 = nn.AvgPool1d(cfgs.p_times, stride=cfgs.p_times)
        self.pool3 = nn.AvgPool1d(cfgs.k_times, stride=cfgs.k_times)

        # compute the score of the question that can be answered correctly
        self.score_lang = MLP(
            dims=[cfgs.hidden_size, cfgs.flat_mlp_size, cfgs.flat_glimpse],
            act='',
            dropout_r=cfgs.dropout_r
        )

        # use ban to fuse features of questions and diagrams
        self.backbone = BAN(cfgs)

        # Classification layers
        layers = [
            weight_norm(nn.Linear(cfgs.hidden_size * 8, cfgs.flat_out_size), dim=None),
            nn.ReLU(),
            nn.Dropout(cfgs.classifer_dropout_r, inplace=True),
            weight_norm(nn.Linear(cfgs.flat_out_size, 1), dim=None)
        ]
        self.classifer = nn.Sequential(*layers)

    def forward(self, que_ix, opt_ix, cp_ix):
        """
        :param que_ix: the index of questions
        :param opt_ix: the index of options
        :param dia: the diagram corresponding the above question
        :param ins_dia: the instructional diagram corresponding to the lesson that contains the above question
        :param cp_ix: the closest paragraph that is extracted by TF-IDF method
        """

        batch_size = que_ix.shape[0]
        que_mask = make_mask(que_ix.unsqueeze(2))
        que_feat = self.embedding(que_ix)
        que_feat, _ = self.encode_lang(que_feat)

        opt_num = make_mask(opt_ix)  # to get the actual number of options
        opt_ix = opt_ix.reshape(-1, self.cfgs.max_opt_token)
        opt_mask = make_mask(opt_ix.unsqueeze(2))

        opt_feat = self.embedding(opt_ix)
        opt_feat, _ = self.encode_lang(opt_feat)
        opt_feat = self.flat(opt_feat, opt_mask)
        opt_feat = opt_feat.reshape(batch_size, self.cfgs.max_ndq_ans, -1)

        cp_ix = cp_ix.reshape(-1, self.cfgs.max_sent_token)
        cp_mask = make_mask(cp_ix.unsqueeze(2))
        cp_feat = self.embedding(cp_ix)
        cp_feat, _ = self.encode_lang(cp_feat)
        cp_feat = self.flat(cp_feat, cp_mask)
        cp_feat = cp_feat.reshape(batch_size, self.cfgs.max_sent, -1)
        span_feat = get_span_feat(cp_feat, self.cfgs.span_width)
        cp_sent_mask = make_mask(cp_mask == False).reshape(batch_size, -1)
        cp_sent_mask = cp_sent_mask.unsqueeze(-1).repeat(1, 1, self.cfgs.span_width).reshape(batch_size, -1)

        # compute the entropy of questions
        flattened_que_feat = self.flat(que_feat, que_mask)
        que_entropy = - self.sigmoid(self.score_lang(flattened_que_feat)) * torch.log2(
            self.sigmoid(self.score_lang(flattened_que_feat))) - \
                      (1 - self.sigmoid(self.score_lang(flattened_que_feat))) * torch.log2(
            1 - self.sigmoid(self.score_lang(flattened_que_feat)))

        # compute the conditional entropy of questions given candidate spans
        span_num = span_feat.shape[1]
        flattened_que_feat_expand = flattened_que_feat.repeat(1, span_num).reshape(batch_size, span_num, -1)
        que_with_evi_feat = torch.cat((flattened_que_feat_expand, span_feat), dim=-1)
        que_with_evi_feat = self.pool3(que_with_evi_feat)
        span_feat = self.pool2(span_feat)

        evi_logit = self.sigmoid(self.score_lang(span_feat))
        que_cond_ent = evi_logit * (- self.sigmoid(self.score_lang(que_with_evi_feat)) * torch.log2(
            self.sigmoid(self.score_lang(que_with_evi_feat))) - (1 - self.sigmoid(
            self.score_lang(que_with_evi_feat))) * torch.log2(
            1 - self.sigmoid(self.score_lang(que_with_evi_feat))))

        # compute the information gains of questions given evidence
        inf_gain = que_entropy - que_cond_ent.squeeze(-1)
        evi_ix = get_ix(inf_gain, cp_sent_mask)
        evi_feat = torch.cat([span_feat[i][int(ix)] for i, ix in enumerate(evi_ix)], dim=0).reshape(batch_size, -1)

        # repeat flattened question features
        flattened_que_feat = flattened_que_feat.repeat(1, self.cfgs.max_ndq_ans).reshape(batch_size, self.cfgs.max_ndq_ans, -1)

        # use ban to fuse features of questions and evi_feat
        fuse_que_evi = self.backbone(que_feat, evi_feat)
        fuse_que_evi = fuse_que_evi.mean(1).repeat(1, self.cfgs.max_ndq_ans).reshape(batch_size, self.cfgs.max_ndq_ans, -1)
        evi_feat = evi_feat.repeat(1, self.cfgs.max_ndq_ans).reshape(batch_size, self.cfgs.max_ndq_ans, -1)

        # fuse features of evidence and options
        fuse_evi_opt = evi_feat * opt_feat

        # fuse features of questions and options
        fuse_que_opt = flattened_que_feat * opt_feat

        # fuse features of questions and evidence
        fuse_flat_que_evi = flattened_que_feat * evi_feat

        # fuse features of questions, options, diagrams and evidence
        fuse_all = flattened_que_feat * opt_feat * fuse_que_evi

        fuse_feat = torch.cat(
            (flattened_que_feat,
             opt_feat,
             evi_feat,
             fuse_flat_que_evi,
             fuse_evi_opt,
             fuse_que_opt,
             fuse_que_evi,
             fuse_all),
            dim=-1
        )

        proj_feat = self.classifer(fuse_feat).squeeze(-1)
        proj_feat = proj_feat.masked_fill(opt_num, -1e9)
        return proj_feat, torch.sum((opt_num == 0), dim=-1)
