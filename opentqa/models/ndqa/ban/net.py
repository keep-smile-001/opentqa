# -----------------------------------------------
# !/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time:2020/8/6 15:30
# -----------------------------------------------
import torch
import torch.nn as nn
from opentqa.models.dqa.hmfn.layer import BAN
from torch.nn.utils.weight_norm import weight_norm
from utils.tools import make_mask


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
        self.lang = nn.GRU(
            input_size=cfgs.word_emb_size,
            hidden_size=cfgs.hidden_size_lang,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

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

    def forward(self, que_ix, opt_ix, cp_ix):
        """
        :param que_ix: the index of questions
        :param opt_ix: the index of options
        :param dia: the diagram corresponding the above question
        :param ins_dia: the instructional diagram corresponding to the lesson that contains the above question
        :param cp_ix: the closest paragraph that is extracted by TF-IDF method
        """
        batch_size = que_ix.shape[0]
        que_feat = self.embedding(que_ix)
        que_feat, _ = self.lang(que_feat)

        opt_num = make_mask(opt_ix)                     # to get the actual number of options
        opt_feat = self.embedding(opt_ix)
        opt_feat = opt_feat.reshape((-1, self.cfgs.max_opt_token, self.cfgs.word_emb_size))
        _, opt_feat = self.lang(opt_feat)
        opt_feat = opt_feat.transpose(1, 0).reshape(self.cfgs.max_ndq_ans * batch_size, -1).reshape(batch_size,
                                                                                                self.cfgs.max_ndq_ans,
                                                                                                -1)
        cp_ix = cp_ix.reshape(-1, self.cfgs.max_sent_token)
        cp_feat = self.embedding(cp_ix)
        cp_feat, _ = self.lang(cp_feat)
        cp_feat = cp_feat.reshape(batch_size, self.cfgs.max_sent * self.cfgs.max_sent_token, -1)

        fusion_feat = self.backbone(que_feat, cp_feat)

        fusion_feat = self.flatten(fusion_feat.sum(1))
        fusion_feat = fusion_feat.repeat(1, self.cfgs.max_ndq_ans).reshape(batch_size, self.cfgs.max_ndq_ans, -1)

        proj_feat = self.classifer(fusion_feat, opt_feat)
        proj_feat = proj_feat.masked_fill(opt_num, -1e9)
        return proj_feat, torch.sum((opt_num == 0), dim=-1)
