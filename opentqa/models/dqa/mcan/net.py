# -----------------------------------------------
# !/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time:2020/11/6 15:30
# @Author:Ma Jie
# -----------------------------------------------


import torch
import torch.nn as nn
from opentqa.models.dqa.mcan.layer import MCA_ED, SimCLR, AttFlat, LayerNorm


class Net(nn.Module):
    """
        :param cfgs: configurations of XTQA.
        :param pretrained_emb:
        :param token_size: the size of vocabulary table.
    """

    def __init__(self, cfgs, pretrained_emb, token_size):
        super(Net, self).__init__()
        self.cfgs = cfgs
        self.answer_size = cfgs.max_ans

        self.embedding = nn.Embedding(
            token_size,
            cfgs.word_emb_size
        )

        if cfgs.pretrained_emb['name']:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_emb))

        # use simclr to encode features of diagrams
        self.simclr = SimCLR(cfgs)

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
        self.proj = nn.Linear(cfgs.dia_feat_size, 1)

        self.dia_linear = nn.Linear(cfgs.dia_feat_size, cfgs.hidden_size)

    def forward(self, que_ix, opt_ix, dia, ins_dia, cp_ix):
        """
        :param que_ix: the index of questions
        :param opt_ix: the index of options
        :param dia: the diagram corresponding the above question
        :param ins_dia: the instructional diagram corresponding to the lesson that contains the above question
        :param cp_ix: the closest paragraph that is extracted by TF-IDF method
        """
        batch_size = que_ix.shape[0]
        dia_feat = self.simclr(dia)
        dia_feat = dia_feat.view(batch_size, -1, self.cfgs.hidden_size)
        lang_feat_mask = self.make_mask(que_ix.unsqueeze(2))
        que_feat = self.embedding(que_ix)
        que_feat, _ = self.lstm(que_feat)
        img_feat_mask = self.make_mask(dia_feat)
        # Backbone Framework
        lang_feat, img_feat = self.backbone(
            que_feat,
            dia_feat,
            lang_feat_mask,
            img_feat_mask
        )
        lang_feat = self.attflat_lang(
            lang_feat,
            lang_feat_mask
        )

        img_feat = self.attflat_img(
            img_feat,
            img_feat_mask
        )

        proj_feat = lang_feat + img_feat
        proj_feat = self.proj_norm(proj_feat)

        opt_ix = opt_ix.reshape(-1, self.cfgs.max_opt_token)
        opt_feat_mask = self.make_mask(opt_ix.unsqueeze(2))
        opt_feat = self.embedding(opt_ix)
        opt_feat, _ = self.lstm(opt_feat)
        opt_feat = self.attflat_lang(
            opt_feat,
            opt_feat_mask
        )
        opt_feat = opt_feat.reshape(batch_size, self.cfgs.max_ans, -1)  # opt_feat:   [batch*4,1024]

        proj_feat = proj_feat.repeat(1, self.cfgs.max_ans).reshape(batch_size, self.cfgs.max_ans,
                                                                   self.cfgs.flat_out_size)
        fuse_feat = torch.cat((proj_feat, opt_feat), dim=-1)
        logits = self.proj(fuse_feat).squeeze(-1)

        return logits

    def make_mask(self, feature):
        return (torch.sum(
            torch.abs(feature),
            dim=-1
        ) == 0).unsqueeze(1).unsqueeze(2)
