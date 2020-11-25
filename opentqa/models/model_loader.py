#-----------------------------------------------
#!/usr/bin/env python
# _*_ coding: utf-8 _*_
#@Time:2020/8/15 16:35
#@Author:Ma Jie
#-----------------------------------------------

from importlib import import_module
from opentqa.core.base_path import PATH


class ModelLoader(object):
    def __init__(self, cfgs):

        self.model_use = cfgs.model
        model_moudle_path = 'opentqa.models.' + cfgs.dataset_use + '.' + self.model_use + '.net'
        self.model_moudle = import_module(model_moudle_path)

    def Net(self, __arg1, __arg2, __arg3):
        return self.model_moudle.Net(__arg1, __arg2, __arg3)


class CfgLoader(object):
    def __init__(self, dataset, model_use):
        cfg_moudle_path = 'opentqa.models.' + dataset + '.' + model_use + '.model_cfgs'
        self.cfg_moudle = import_module(cfg_moudle_path)

    def load(self):
        return self.cfg_moudle.Cfgs()
