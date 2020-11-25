# -----------------------------------------------
# !/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time:2020/8/6 15:30
# @Author:Ma Jie
# -----------------------------------------------
import torch
import torch.nn as nn
from opentqa.models.dqa.hmfn.layer import SimCLR, HMFN_V1, BAN, FlattenAtt
from utils.tools import get_related_diagram, make_mask
from torch.nn.utils.weight_norm import weight_norm


class Net(nn.Module):
    def __init__(self, cfgs, pretrained_emb, token_size):
        """
        :param cfgs: configurations of XTQA.
        :param pretrained_emb:
        :param token_size: the size of vocabulary table.
        """
        super(Net, self).__init__()
        self.cfgs = cfgs
        self.embedding = nn.Embedding(
            num_embeddings=token_size,
            embedding_dim=cfgs.word_emb_size
        )
        if cfgs.pretrained_emb['name']:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_emb))

        # use rnn to encode the closest paragraph.
        self.rnn_cp = nn.GRU(
            input_size=cfgs.word_emb_size,
            hidden_size=cfgs.hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        # use rnn to encode the question and option
        self.rnn_qo = nn.GRU(
            input_size=cfgs.word_emb_size,
            hidden_size=cfgs.hidden_size_qo,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.flatten_opt = FlattenAtt(cfgs)

        # use simclr to encode the diagram
        self.simclr = SimCLR(cfgs)

        # hierarchical bilinear attention networks
        self.backbone = HMFN_V1(cfgs)

        # Note: if use this strategy, you must set the batch size=1
        if cfgs.divide_and_rule in 'True':
            # predict whether this question needs multimodal context
            self.classifer0 = weight_norm(nn.Linear(cfgs.hidden_size, 1), dim=None)
            self.sigmoid = nn.Sigmoid()
            self.ban = BAN(cfgs)

        # Classification layers
        layers = [
            weight_norm(nn.Linear(cfgs.hidden_size, cfgs.flat_out_size), dim=None),
            nn.ReLU(),
            nn.Dropout(cfgs.classifer_dropout_r, inplace=True),
            # weight_norm(nn.Linear(cfgs.flat_out_size, 1), dim=None)
        ]
        self.flatten = nn.Sequential(*layers)
        self.classifer1 = nn.CosineSimilarity(dim=2)

    def forward(self, que_ix, opt_ix, dia, ins_dia, cp_ix):
        """
        :param que_ix: the index of questions
        :param opt_ix: the index of options
        :param dia: the diagram corresponding the above question
        :param ins_dia: the instructional diagram corresponding to the lesson that contains the above question
        :param cp_ix: the closest paragraph that is extracted by TF-IDF method
        """
        batch_size = que_ix.shape[0]
        que_feat = self.embedding(que_ix)
        que_feat, que_hidden = self.rnn_qo(que_feat)

        # opt_feat = self.embedding(opt_ix)
        # opt_feat = opt_feat.reshape((-1, self.cfgs.max_opt_token, self.cfgs.word_emb_size))
        # _, opt_feat = self.rnn_qo(opt_feat)
        # opt_feat = opt_feat.transpose(1, 0).reshape(self.cfgs.max_dq_ans * batch_size, -1).reshape(batch_size,
        #                                                                                         self.cfgs.max_dq_ans,
        #                                                                                         -1)
        opt_ix = opt_ix.reshape(-1, self.cfgs.max_opt_token)
        opt_mask = make_mask(opt_ix.unsqueeze(2))
        opt_feat = self.embedding(opt_ix)
        opt_feat, _ = self.rnn_qo(opt_feat)
        opt_feat = self.flatten_opt(opt_feat, opt_mask)
        opt_feat = opt_feat.reshape(batch_size, self.cfgs.max_dq_ans, -1)

        cp_feat = self.embedding(cp_ix)
        cp_feat, _ = self.rnn_cp(cp_feat.reshape(batch_size, -1, self.cfgs.word_emb_size))

        dia_feat = self.simclr(dia)
        dia_feat = dia_feat.reshape(batch_size, -1)

        # if self.cfgs.use_ins_dia in 'True':
        #     ins_dia_feat = ins_dia.reshape(-1, 3, self.cfgs.input_size, self.cfgs.input_size)
        #     ins_dia_feat = self.simclr(ins_dia_feat)
        #     ins_dia_feat = ins_dia_feat.reshape(batch_size, -1, self.cfgs.dia_feat_size)
        #     related_ins_dia_feat = get_related_diagram(dia_feat, ins_dia_feat)
        #     dia_feat = (dia_feat + related_ins_dia_feat) / 2.0

        # fusion_feat = self.backbone(que_opt_feat, dia_feat, cp_feat)
        fusion_feat = self.forward_with_divide_and_rule(que_feat, dia_feat, ins_dia, cp_feat, que_hidden)
        fusion_feat = self.flatten(fusion_feat.sum(1))
        fusion_feat = fusion_feat.repeat(1, self.cfgs.max_dq_ans).reshape(batch_size, self.cfgs.max_dq_ans, -1)
        proj_feat = self.classifer1(fusion_feat, opt_feat)

        return proj_feat

    # whether use the divide-and-rule strategy
    def forward_with_divide_and_rule(self, que_feat, dia_feat, ins_dia, cp_feat, que_hidden):
        batch_size = que_feat.shape[0]
        logits = 10

        if self.cfgs.divide_and_rule in 'True':
            que_hidden = que_hidden.transpose(1, 0).reshape(self.cfgs.max_dq_ans, -1).reshape(batch_size,
                                                                                                   self.cfgs.max_dq_ans,
                                                                                                   -1)
            logits = self.sigmoid(self.classifer0(que_hidden))

        if logits < 0.5:
            feat_dim = dia_feat.shape[-1]
            dia_feat = dia_feat.reshape(batch_size, -1, feat_dim)
            fusion_feat = self.ban(que_feat, dia_feat)
        else:
            if self.cfgs.use_ins_dia in 'True':
                ins_dia_feat = ins_dia.reshape(-1, 3, self.cfgs.input_size, self.cfgs.input_size)
                ins_dia_feat = self.simclr(ins_dia_feat)
                ins_dia_feat = ins_dia_feat.reshape(batch_size, -1, self.cfgs.dia_feat_size)
                related_ins_dia_feat = get_related_diagram(dia_feat, ins_dia_feat)
                dia_feat = (dia_feat + related_ins_dia_feat) / 2.0

            fusion_feat = self.backbone(que_feat, dia_feat, cp_feat)
        return fusion_feat