# -----------------------------------------------
# !/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time:2020/11/6 15:30
# -----------------------------------------------


import torch
import torch.nn as nn
from opentqa.models.dqa.mcan.layer import MCA_ED, AttFlat, LayerNorm
from utils.tools import make_mask


class Net(nn.Module):
    """
        :param cfgs: configurations of XTQA.
        :param pretrained_emb:
        :param token_size: the size of vocabulary table.
    """

    def __init__(self, cfgs, pretrained_emb, token_size):
        super(Net, self).__init__()
        self.cfgs = cfgs
        self.answer_size = cfgs.max_ndq_ans

        self.embedding = nn.Embedding(
            token_size,
            cfgs.word_emb_size
        )

        if cfgs.pretrained_emb['name']:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_emb))

        self.lstm = nn.LSTM(
            input_size=cfgs.word_emb_size,
            hidden_size=cfgs.hidden_size,
            num_layers=1,
            batch_first=True
        )

        self.backbone = MCA_ED(cfgs)
        self.attflat_img = AttFlat(cfgs)
        self.attflat_lang = AttFlat(cfgs)

        self.proj_norm = LayerNorm(cfgs.flat_out_size)
        self.proj = nn.Linear(cfgs.cp_feat_size, 1)

    def forward(self, que_ix, opt_ix, cp_ix):
        """
        :param que_ix: the index of questions
        :param opt_ix: the index of options
        :param dia: the diagram corresponding the above question
        :param ins_dia: the instructional diagram corresponding to the lesson that contains the above question
        :param cp_ix: the closest paragraph that is extracted by TF-IDF method
        """
        batch_size = que_ix.shape[0]
        lang_feat_mask = self.make_mask(que_ix.unsqueeze(2))
        que_feat = self.embedding(que_ix)
        que_feat, _ = self.lstm(que_feat)

        opt_num = make_mask(opt_ix)  # to get the actual number of options
        opt_ix = opt_ix.reshape(-1, self.cfgs.max_opt_token)
        opt_feat_mask = self.make_mask(opt_ix.unsqueeze(2))
        opt_feat = self.embedding(opt_ix)
        opt_feat, _ = self.lstm(opt_feat)

        opt_feat = self.attflat_lang(
            opt_feat,
            opt_feat_mask
        )
        opt_feat = opt_feat.reshape(batch_size, self.cfgs.max_ndq_ans, -1)  # opt_feat:   [batch*4,1024]

        cp_ix = cp_ix.reshape(batch_size, -1)
        cp_mask = make_mask(cp_ix.unsqueeze(2))
        cp_feat = self.embedding(cp_ix)
        cp_feat, _ = self.lstm(cp_feat)

        # Backbone Framework
        lang_feat, cp_feat = self.backbone(
            que_feat,
            cp_feat,
            lang_feat_mask,
            cp_mask
        )
        lang_feat = self.attflat_lang(
            lang_feat,
            lang_feat_mask
        )

        cp_feat = self.attflat_img(
            cp_feat,
            cp_mask
        )

        proj_feat = lang_feat + cp_feat
        proj_feat = self.proj_norm(proj_feat)

        proj_feat = proj_feat.repeat(1, self.cfgs.max_ndq_ans).reshape(batch_size, self.cfgs.max_ndq_ans,
                                                                   self.cfgs.flat_out_size)
        fuse_feat = torch.cat((proj_feat, opt_feat), dim=-1)
        proj_feat = self.proj(fuse_feat).squeeze(-1)

        proj_feat = proj_feat.masked_fill(opt_num, -1e9)
        return proj_feat, torch.sum((opt_num == 0), dim=-1)

    def make_mask(self, feature):
        return (torch.sum(
            torch.abs(feature),
            dim=-1
        ) == 0).unsqueeze(1).unsqueeze(2)
