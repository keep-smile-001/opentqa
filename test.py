#-----------------------------------------------
#!/usr/bin/env python
# _*_ coding: utf-8 _*_
#@Time:2020/8/3 19:28
#-----------------------------------------------
import re
import torch
import numpy as np

# v = torch.randn(64, 100, 2048)
# mask = (0 == v.abs().sum(2)).unsqueeze(1).unsqueeze(3)
# print(mask.shape)
# a = np.random.rand(5)
# b = np.random.rand(5)
# c = []
#
# d = a == b
# print(d.sum())
# a = 'True'
#
# if a is 'True':
#     print('--')

# a = torch.randn(2, 1, 6)
# print(a)
#
# b = a.repeat(1, 4, 1).reshape(2, 4, 1, 6)
# print(b)

# a = torch.randn(1, 4, 4)
# b = torch.mean(a, dim=1)
# print(b.shape)

# import spacy
# nlp = spacy.load('en_core_web_sm')
#
# txt = "some text read from one paper ..."
# doc = nlp(txt)
#
# for sent in doc.sents:
#     print(sent)
#     print('#'*50)
from utils.tools import make_mask
#
# a = torch.randn((1, 5, 10))
# print(a)
#
#
# start = torch.arange(0, 5).unsqueeze(-1).repeat(1, 2)
# print(start)
# end = torch.arange(0, 2).unsqueeze(0) + start
# print(end)
# mask = (end < 5)
# print(mask)
#
#
# flatten_start = torch.masked_select(start, mask)
# print(flatten_start.shape)
#
# flatten_end = torch.masked_select(end, mask)
# print(flatten_end)
#
# print('-----------------')
# print(flatten_start)
# span_start_emb = a[:, flatten_start, :]
# print(span_start_emb)

# a = torch.randn((2, 19, 2048))
# b = torch.arange(0, 2).unsqueeze(-1)
# print(b)
# c = torch.index_select(a, dim=0, index=b)
# print(c.shape)


# a = torch.arange(1, 8).reshape(1, 7)
# b = a[:, :-1]
# print(b)
# from torch.optim.adam import Adam
#
# optimizer = Adam(weight_decay=1e-6)
# contents = 'Overall Accuracy is {} -> [TF Accuracy is {} || MC Accuracy is {}]'.format(str(0.2564569875654), str(0.2564569875654), str(0.2564569875654))
# print(contents + '\n')

a = torch.randn((2, 15, 15))
print(a)
b = torch.sum(a, dim=-1)
c = b.unsqueeze(2)
print(c.shape)