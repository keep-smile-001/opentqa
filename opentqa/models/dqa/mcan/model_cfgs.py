# -----------------------------------------------
# !/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time:2020/10/19 9:59
# -----------------------------------------------

from opentqa.core.base_cfgs import Configurations


class Cfgs(Configurations):
    def __init__(self):
        super(Cfgs, self).__init__()
        self.max_ans = 4            # for diagram questions, max_ans=4; for non-diagram questions, max_ans=7.
        self.hidden_size = 512
        self.dropout_r = 0.1
        self.dia_feat_size = 2048   # the feature of the diagram
        self.multi_head = 8         # Multi-head number in MCA layers
        # (Warning: HIDDEN_SIZE should be divided by MULTI_HEAD)
        self.layer = 6              # Model deeps
        # (Encoder and Decoder will be same deeps)
        self.flat_mlp_size = 512    # MLP size in flatten layers
        self.flat_glimpses = 1      # Flatten the last hidden to vector with {n} attention glimpses
        self.flat_out_size = 1024
        self.ff_size = int(self.hidden_size * 4)
        self.hidden_size_head = int(self.hidden_size / self.multi_head)
