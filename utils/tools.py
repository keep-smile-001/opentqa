# -----------------------------------------------
# !/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time:2020/9/16 22:02
# @Author:Ma Jie
# -----------------------------------------------
import torch
import torch.nn.functional as F


def get_related_diagram(dia_feat, ins_dia_feat):
    '''
    compute the similarity between a diagram and its corresponding instructional diagrams
    :param dia_feat: [b, d]
    :param ins_dia_feat:[b, num_ins, d]
    :return: the feature of the most related instructional diagrams
    '''
    batch_size = dia_feat.shape[0]
    num_ins_dia = ins_dia_feat.shape[1]

    dia_feat_expand = dia_feat.repeat(1, num_ins_dia).reshape(batch_size, num_ins_dia, -1)
    sim = F.cosine_similarity(dia_feat_expand, ins_dia_feat, dim=-1)
    max_sim, ix_arr = torch.max(sim, dim=-1) # ix_arr: [1, batch_size]

    related_ins_dia_feat = torch.cat([ins_dia_feat[i][ix] for i, ix in enumerate(ix_arr)], dim=0).reshape(batch_size, -1)
    return related_ins_dia_feat

def get_span_feat(input, span_width):
    """
    :param input: [batch_size, max_sent, dim] tensor
    :param span_width: width of spans
    :return:
    """
    max_sent = input.shape[1]

    start = torch.arange(0, max_sent).unsqueeze(-1).repeat(1, span_width)
    end = torch.arange(0, span_width).unsqueeze(0) + start

    sent_mask = (end < max_sent)
    flattened_start = torch.masked_select(start, sent_mask)
    flattened_end = torch.masked_select(end, sent_mask)

    span_start_feat = input[:, flattened_start, :]
    span_end_feat =  input[:, flattened_end, :]
    span_feat = torch.cat((span_start_feat, span_end_feat), -1)

    return span_feat

# get the index of the maximum value
def get_ix(input, mask):
    mask = mask[:, :-1]
    input = input.masked_fill(mask, -1e9)
    _, ix = torch.max(input, dim=-1)
    return ix

# mask the input and return a boolean tensor
def make_mask(input):
    return (torch.sum(torch.abs(input), dim=-1) == 0)

# write some information into a log
def write_log(cfgs, content):
    log_file = open(
        cfgs.log_path + '/log_' +
        cfgs.dataset_use + '_' + cfgs.model + '_' +
        cfgs.run_mode + '_' +
        cfgs.version + '.txt', 'a+')

    log_file.write(str(content))
    log_file.close()

