# -----------------------------------------------
# !/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time:2020/9/25 8:28
# -----------------------------------------------
import torch.optim as Optim


class WarmupOptimizer(object):
    def __init__(self, lr_base, optimizer, data_size, batch_size, warmup_epoch):
        self.optimizer = optimizer
        self._step = 0
        self.lr_base = lr_base
        self._rate = 0
        self.data_size = data_size
        self.batch_size = batch_size
        self.warmup_epoch = warmup_epoch

    def step(self):
        self._step += 1

        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate

        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def rate(self, step=None):
        if step is None:
            step = self._step

        if step <= int(self.data_size / self.batch_size * (self.warmup_epoch + 1) * 0.25):
            r = self.lr_base * 1 / (self.warmup_epoch + 1)
        elif step <= int(self.data_size / self.batch_size * (self.warmup_epoch + 1) * 0.5):
            r = self.lr_base * 2 / (self.warmup_epoch + 1)
        elif step <= int(self.data_size / self.batch_size * (self.warmup_epoch + 1) * 0.75):
            r = self.lr_base * 3 / (self.warmup_epoch + 1)
        else:
            r = self.lr_base

        return r


def get_optim(cfgs, model, data_size, lr_base=None):
    if lr_base is None:
        lr_base = cfgs.lr_base

    std_optim = getattr(Optim, cfgs.opt)
    params = filter(lambda p: p.requires_grad, model.parameters())
    eval_str = 'params, lr=0'
    for key in cfgs.opt_params:
        eval_str += ' ,' + key + '=' + str(cfgs.opt_params[key])

    optim = WarmupOptimizer(
        lr_base,
        eval('std_optim' + '(' + eval_str + ')'),
        data_size,
        int(cfgs.batch_size),
        cfgs.warmup_epoch
    )

    return optim


def adjust_lr(optim, decay_r):
    optim.lr_base *= decay_r
