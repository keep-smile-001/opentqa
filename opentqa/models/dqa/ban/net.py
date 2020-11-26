# -----------------------------------------------
# !/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time:2020/8/6 15:30
# -----------------------------------------------
import torch
import torch.nn as nn
from opentqa.models.dqa.hmfn.layer import SimCLR, BAN
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

        # use rnn to encode the question and option
        self.rnn_qo = nn.GRU(
            input_size=cfgs.word_emb_size,
            hidden_size=cfgs.hidden_size_qo,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        # use simclr to encode the diagram
        self.simclr = SimCLR(cfgs)

        # bilinear attention networks
        self.backbone = BAN(cfgs)

        # Classification layers
        layers = [
            weight_norm(nn.Linear(cfgs.hidden_size, cfgs.flat_out_size), dim=None),
            nn.ReLU(),
            nn.Dropout(cfgs.classifer_dropout_r, inplace=True),
            # weight_norm(nn.Linear(cfgs.flat_out_size, 1), dim=None)
        ]
        self.flatten = nn.Sequential(*layers)
        self.classifer = nn.CosineSimilarity(dim=2)

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
        que_feat, _ = self.rnn_qo(que_feat)

        opt_feat = self.embedding(opt_ix)
        opt_feat = opt_feat.reshape((-1, self.cfgs.max_opt_token, self.cfgs.word_emb_size))
        _, opt_feat = self.rnn_qo(opt_feat)
        opt_feat = opt_feat.transpose(1, 0).reshape(self.cfgs.max_dq_ans * batch_size, -1).reshape(batch_size,
                                                                                                self.cfgs.max_dq_ans,
                                                                                                -1)

        dia_feat = self.simclr(dia)
        dia_feat = dia_feat.reshape(batch_size, -1)

        dia_feat_dim = dia_feat.shape[-1]
        dia_feat = dia_feat.reshape(batch_size, -1, dia_feat_dim)
        fusion_feat = self.backbone(que_feat, dia_feat)

        fusion_feat = self.flatten(fusion_feat.sum(1))
        fusion_feat = fusion_feat.repeat(1, self.cfgs.max_dq_ans).reshape(batch_size, self.cfgs.max_dq_ans, -1)
        proj_feat = self.classifer(fusion_feat, opt_feat)

        return proj_feat
