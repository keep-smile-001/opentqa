#-----------------------------------------------
#!/usr/bin/env python
# _*_ coding: utf-8 _*_
#@Time:2020/9/14 11:08
#@Author:Ma Jie
#-----------------------------------------------
from importlib import import_module

class DatasetLoader(object):
    def __init__(self, cfgs, split):
        self.cfgs = cfgs
        self.split = split
        dataset_moudle_path = 'opentqa.datasets.' + self.cfgs.dataset_use + '_loader'
        self.dataset_moudle = import_module(dataset_moudle_path)

    def dataset(self):
        return self.dataset_moudle.DataSet(self.cfgs, self.split)