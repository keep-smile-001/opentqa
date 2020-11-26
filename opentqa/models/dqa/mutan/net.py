# -----------------------------------------------
# !/usr/bin/env python
# _*_coding:utf-8 _*_
# @Time: 2020/11/1 17:19
# @File : net.py
# -----------------------------------------------
# -----------------------------------------------
# !/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time:2020/8/6 15:30
# -----------------------------------------------
import torch
import torch.nn as nn

from opentqa.models.dqa.MUTAN.layer import MuTAN, FlattenAtt
from opentqa.models.dqa.ban.layer import SimCLR
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
        self.encode_lang = nn.GRU(
            input_size=cfgs.word_emb_size,
            hidden_size=cfgs.hidden_size,
            batch_first=True,
        )
        # use simclr to encode the diagram
        self.simclr = SimCLR(cfgs)
        self.flat = FlattenAtt(cfgs)


        # bilinear attention networks
        self.backbone = MuTAN(v_relation_dim=self.cfgs.relation_dim, num_hid=self.cfgs.hidden_size, num_ans_candidates=cfgs.num_ans_candidates,
                              gamma=cfgs.mutan_gamma)

        # Classification layers
        layers = [
            weight_norm(nn.Linear(cfgs.hidden_size*5, cfgs.flat_out_size), dim=None),
            nn.ReLU(),
            nn.Dropout(cfgs.classifer_dropout_r, inplace=True),
            weight_norm(nn.Linear(cfgs.flat_out_size, 1), dim=None)
        ]
        self.classifer = nn.Sequential(*layers)
        # self.classifer = nn.CosineSimilarity(dim=2)

    def forward(self, que_ix, opt_ix, dia, dia_node_ix, ins_dia, cp_ix):
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
        que_feat = self.flat(que_feat, que_mask)

        opt_ix = opt_ix.reshape(-1, self.cfgs.max_opt_token)
        opt_mask = make_mask(opt_ix.unsqueeze(2))

        opt_feat = self.embedding(opt_ix)
        opt_feat, _ = self.encode_lang(opt_feat)
        opt_feat = self.flat(opt_feat, opt_mask)
        opt_feat = opt_feat.reshape(batch_size, self.cfgs.max_dq_ans, -1)

        dia_feat = self.simclr(dia)
        dia_feat = dia_feat.reshape(batch_size, -1)

        dia_feat_dim = dia_feat.shape[-1]
        dia_feat = dia_feat.reshape(batch_size, 1, dia_feat_dim)
        fusion_feat = self.backbone(dia_feat, que_feat) #[8,4096]  BAN得到的是[b,15,1024]

        # fusion_feat = self.flatten(fusion_feat.sum(1))
        fusion_feat = fusion_feat.repeat(1, self.cfgs.max_dq_ans).reshape(batch_size, self.cfgs.max_dq_ans, -1)
        fuse_opt_feat = torch.cat((fusion_feat, opt_feat), dim=-1) #(8,4,4096)(8,4,1024)
        proj_feat = self.classifer(fuse_opt_feat).squeeze(-1)

        return proj_feat
