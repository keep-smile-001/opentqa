#-----------------------------------------------
#!/usr/bin/env python
# _*_ coding: utf-8 _*_
#@Time:2020/8/15 17:39
#@Author:Ma Jie
#-----------------------------------------------
from opentqa.core.base_cfgs import Configurations

class Cfgs(Configurations):
    def __init__(self):
        super(Cfgs, self).__init__()
        self.classifer_dropout_r = 0.2
        self.dropout_r = 0.2
        self.cp_feat_size = 1024        # the feature of the diagram, here it denotes the dimension of the closest paragraph.
        self.flat_out_size = 1024       # flatten hidden size
        self.glimpse = 1
        self.hidden_size = 1024
        self.hidden_size_lang = 512       # the hidden size of the closest paragraph
        self.k_times = 3
        self.max_ans = 4                # for diagram questions, max_ans=4; for non-diagram questions, max_ans=7.
        self.ba_hidden_size = self.k_times * self.hidden_size
