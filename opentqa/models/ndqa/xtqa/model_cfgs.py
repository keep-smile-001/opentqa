#-----------------------------------------------
#!/usr/bin/env python
# _*_ coding: utf-8 _*_
#@Time:2020/10/19 9:59
#@Author:Ma Jie
#-----------------------------------------------

from opentqa.core.base_cfgs import Configurations

class Cfgs(Configurations):
    def __init__(self):
        super(Cfgs, self).__init__()
        self.classifer_dropout_r = 0.1
        self.dropout_r = 0.1
        self.dia_feat_size = 1024       # the feature of the diagram, Note in ndqa this means the feature of evidence
        self.flat_out_size = 2048       # flatten hidden size
        self.flat_mlp_size = 512        # flatten one tensor to 512-dimensional tensor
        self.flat_glimpse = 1
        self.glimpse = 1
        self.hidden_size = 1024
        self.lr_base = 0.0001
        self.lr_decay_r = 0.15          # decay rate test:0.1 better;
        self.k_times = 3
        self.max_opt_token = 5          # the maximum token of a option
        self.max_que_token = 20         # the maximum token of a [question] test: 10; better;
        self.max_sent = 15              # the maximum number of sentences within a paragraph test: 15 better;
        self.max_sent_token = 15        # the maximum number of tokens per sentence; test:15 better;    # for diagram questions, max_ans=4; for non-diagram questions, max_ans=7.
        self.ba_hidden_size = self.k_times * self.hidden_size
        self.span_width = 2
        self.p_times = 2