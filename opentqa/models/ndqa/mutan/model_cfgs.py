# -----------------------------------------------
# !/usr/bin/env python
# _*_coding:utf-8 _*_
# @Time: 2020/11/1 17:18
# @Author:wyx
# @File : model_cfgs.py
# -----------------------------------------------

from opentqa.core.base_cfgs import Configurations

class Cfgs(Configurations):
    def __init__(self):
        super(Cfgs, self).__init__()
        self.classifer_dropout_r = 0.5
        self.dropout_r = 0.2
        self.dia_feat_size = 2048       # the feature of the diagram
        self.flat_out_size = 1024       # flatten hidden size
        self.flat_mlp_size = 512        # flatten one tensor to 512-dimensional tensor
        self.flat_glimpse = 1
        self.glimpse = 1
        self.hidden_size = 1024
        self.k_times = 3
        self.max_ans = 4                # for diagram questions, max_ans=4; for non-diagram questions, max_ans=7.
        self.ba_hidden_size = self.k_times * self.hidden_size
        self.span_width = 2
        self.p_times = 2

        self.relation_dim = 2048
        self.mutan_gamma = 2
        self.fusion = "mutan"
        self.num_ans_candidates = 4
