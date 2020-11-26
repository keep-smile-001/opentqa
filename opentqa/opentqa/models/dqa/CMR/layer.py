# -----------------------------------------------
# !/usr/bin/env python
# _*_ coding: utf-8 _*_
#------------------------------------------------


import os, torch
import torch.nn as nn
import torch.nn.functional as F
import importlib.util
import math
from torch.nn.utils.weight_norm import weight_norm
from BERT_related.modeling import VISUAL_CONFIG, BertPreTrainedModel
from BERT_related.modeling import CrossEncoder


class SimCLR(nn.Module):
    '''
    use simclr to obtain diagram representations.
    '''

    def __init__(self, cfgs):
        super(SimCLR, self).__init__()
        self.cfgs = cfgs

        spec = importlib.util.spec_from_file_location(
            "simclr",
            os.path.join(self.cfgs.simclr['checkpoints_folder'], 'resnet_simclr.py'))

        resnet_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(resnet_module)
        self.simclr_resnet = resnet_module.ResNetSimCLR(self.cfgs.simclr['base_model'],
                                                        int(self.cfgs.simclr['out_dim']))
        # map_location must include 'cuda:0', otherwise it will occur error
        state_dict = torch.load(os.path.join(self.cfgs.simclr['checkpoints_folder'], 'model.pth'),
                                map_location='cuda:{}'.format(self.cfgs.gpu) if torch.cuda.is_available() else 'cpu')
        self.simclr_resnet.load_state_dict(state_dict)

        if self.cfgs.not_fine_tuned in 'True':
            for param in self.simclr_resnet.parameters():
                param.requires_grad = False

    def forward(self, dia):
        dia_feats, _ = self.simclr_resnet(dia)
        return dia_feats

def set_visual_config(cfg):
    VISUAL_CONFIG.l_layers = 9
    VISUAL_CONFIG.x_layers = 5
    VISUAL_CONFIG.r_layers = 5

def convert_sents_to_features(sents, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    for (i, sent) in enumerate(sents):
        tokens_a = tokenizer.tokenize(sent.strip())

        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:(max_seq_length - 2)]

        # Keep segment id which allows loading BERT-weights.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids))
    return features
class BertPooler(nn.Module):
    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        #print("-----pool----\n",first_token_tensor.size(),hidden_states.size())
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output
class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
class BertModel(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.encoder = CrossEncoder(config)
        self.pooler = BertPooler(config)
        self.apply(self.init_bert_weights)

    def forward(self, lang_feats,visn_feats,attention_mask=None):
        if attention_mask is None:
            attention_mask = torch.ones(lang_feats.size()[0:-1]).cuda()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        #print(lang_feats.size(),extended_attention_mask.size())
        lang_feats, visn_feats = self.encoder(
            lang_feats,extended_attention_mask,
            visn_feats=visn_feats)
        #print("--------------------------------------",lang_feats,lang_feats.size())
        pooled_output = self.pooler(lang_feats)

        return (lang_feats, visn_feats), pooled_output
class VisBert(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.apply(self.init_bert_weights)

    def forward(self, sents, visual_feats,
                visual_attention_mask=None):
        feat_seq, pooled_output = self.bert(lang_feats=sents,visn_feats=visual_feats)
        return feat_seq, pooled_output
class BertEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.max_seq_length = cfg.max_seq_length
        set_visual_config(cfg)

        self.model = VisBert.from_pretrained("bert-base-uncased")

        if cfg.from_scratch:
            #print("initializing all the weights")
            self.model.apply(self.model.init_bert_weights)

    def multi_gpu(self):
        self.model = nn.DataParallel(self.model)

    @property
    def dim(self):
        return 768

    def forward(self, sents, feats, visual_attention_mask=None):
        (output_lang, output_img), output_cross = self.model(sents=sents,
                            visual_feats=feats)

        return output_lang, output_img, output_cross
class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        try:
            s = (x - u).pow(2).mean(-1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        except RuntimeError:
            #print("BLN",x.size())
            raise AssertionError
        return self.weight * x + self.bias
class FlattenAtt(nn.Module):
    def __init__(self, cfgs):
        super(FlattenAtt, self).__init__()
        self.cfgs = cfgs

        self.mlp = MLP(
            dims=[cfgs.hidden_size, cfgs.flat_mlp_size, cfgs.flat_glimpse],
            act='',
            dropout_r=cfgs.dropout_r
        )

        self.linear_merge = weight_norm(
            nn.Linear(cfgs.hidden_size * cfgs.flat_glimpse, cfgs.hidden_size),
            dim=None
        )

    def forward(self, x, x_mask):
        att = self.mlp(x)
        att = att.masked_fill(
            x_mask.unsqueeze(2),
            -1e9
        )
        att = F.softmax(att, dim=1)

        att_list = []
        for i in range(self.cfgs.flat_glimpse):
            att_list.append(
                torch.sum(att[:, :, i: i + 1] * x, dim=1)
            )

        x_atted = torch.cat(att_list, dim=1)
        x_atted = self.linear_merge(x_atted)

        return x_atted
class MLP(nn.Module):
    """
    Simple class for non-linear fully connect network
    """

    def __init__(self, dims, act='ReLU', dropout_r=0.0):
        super(MLP, self).__init__()

        layers = []
        for i in range(len(dims) - 1):
            in_dim = dims[i]
            out_dim = dims[i + 1]
            if dropout_r > 0:
                layers.append(nn.Dropout(dropout_r))
            layers.append(weight_norm(nn.Linear(in_dim, out_dim), dim=None))
            if act != '':
                layers.append(getattr(nn, act)())

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)
def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    try:
        _ = x * 0.5
        _ = _ * (1.0 + torch.erf(x / math.sqrt(2.0)))
        return _
    except RuntimeError:
        #print("gelu:",x.size())
        raise AssertionError


class GeLU(nn.Module):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    def __init__(self):
        super().__init__()
        self.count=0
    def forward(self, x):
        self.count+=1
        #print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n",self.count)
        #print(x.size())
        return gelu(x)