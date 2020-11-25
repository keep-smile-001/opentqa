# -----------------------------------------------
# !/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time:2020/8/15 17:07
# @Author:Ma Jie
# -----------------------------------------------

import os

class PATH(object):
    def __init__(self):
        self.init_path()

    def init_path(self):
        self.ckpts_path = './ckpts'                 # the path of checkpoints
        self.log_path = './results/log'             # log file

        if 'results' not in os.listdir('./'):
            os.mkdir('./results')

        if 'log' not in os.listdir('./results'):
            os.mkdir('./results/log')

        if 'ckpts' not in os.listdir('./'):
            os.mkdir('./ckpts')