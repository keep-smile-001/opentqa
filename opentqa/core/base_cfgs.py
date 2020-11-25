# -----------------------------------------------
# !/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time:2020/8/15 17:28
# @Author:Ma Jie
# -----------------------------------------------
import torch, random, os
import numpy as np
from types import MethodType
from opentqa.core.base_path import PATH


class Configurations(PATH):
    def __init__(self):
        super(Configurations, self).__init__()
        self.ckpt_epoch = ''            # use a specified epoch to test models
        self.gpu = '1'
        self.flat_out_size = 2048       # flatten hidden size
        self.grad_norm_clip = -1        # clip the gradient
        self.lr_base = 0.0001           # base learning rate
        self.lr_decay_r = 0.2           # decay rate
        self.lr_decay_list = [8, 10]    # decay epoch list
        self.max_que_token = 15         # the maximum token of a [question]
        self.max_opt_token = 5          # the maximum token of a option
        self.max_sent = 10              # the maximum number of sentences within a paragraph
        self.max_sent_token = 20        # the maximum number of tokens per sentence
        self.max_ins_dia = 5            # the maximum number of the instructional diagram
        self.max_dq_ans = 4             # for diagram questions, max_ans=4; for non-diagram questions, max_ans=7.
        self.max_ndq_ans = 7
        self.input_size = 224           # the size of a diagram
        self.opt = 'Adam'               # optimizer
        self.opt_params = {'betas': '(0.9, 0.98)', 'eps': '1e-9'}       # set the parameters of the optimizer
        self.reduction = 'mean'         # mean, sum, none
        self.seed = 666                 # fix torch and numpy seed
        self.version = str(random.randint(0, 9999))
        self.word_emb_size = 768        # the size of the word embeddings
        self.warmup_epoch = 3           # from the (warm up epoch + 1), the lr is decayed.



    def __str__(self):
        """
        :return: a description of one object.
        """
        __str_object = ""

        for arg in dir(self):
            if not arg.startswith("__") and not isinstance(getattr(self, arg), MethodType):
                __str_object += "{ %-15s } -> " % arg + str(getattr(self, arg)) + '\n'

        return __str_object

    def add_attr(self, args_dict):
        """
        add a specific argument to an object.
        :param args_dict: a dictionary of args.
        :return:
        """
        for arg in args_dict:
            setattr(self, arg, args_dict[arg])

    def proc(self):
        # ------------ Devices setup
        os.environ['CUDA_VISIBLE_DEVICES'] = self.gpu
        self.n_gpu = len(self.gpu.split(','))
        self.devices = [_ for _ in range(self.n_gpu)]
        torch.set_num_threads(2)

        # ------------ seed setup
        # fix pytorch seed
        torch.manual_seed(self.seed)
        if self.n_gpu < 2:
            torch.cuda.manual_seed(self.seed)
        else:
            torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = True

        # fix numpy seed
        np.random.seed(self.seed)

        # fix random seed
        random.seed(self.seed)

    def parse_to_dict(self, args):
        args_dict = {}
        for arg in dir(args):
            if not arg.startswith('_') and not isinstance(getattr(args, arg), MethodType):
                if getattr(args, arg) is not None:
                    args_dict[arg] = getattr(args, arg)
        return args_dict