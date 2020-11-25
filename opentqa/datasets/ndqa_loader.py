# -----------------------------------------------
# !/usr/bin/env python
# _*_coding:utf-8 _*_
# @Time:2020/8/3 9:09
# @Author:Ma Jie
# -----------------------------------------------
import os, re, json, torch, importlib, spacy
import torch.utils.data as Data
import numpy as np
from tqdm import tqdm

nlp = spacy.load('en_core_web_sm')


class DataSet(Data.Dataset):
    def __init__(self, cfgs, split):
        """
        :param cfgs: configuration of opentqa. "en_vectors_web_lg" is the pretrained model of GloVe. Please see the details on "https://spacy.io/models/en-starters#en_vectors_web_lg"
        "en_trf_robertabase_lg" is the pretrained model of BERT. please the details on "https://spacy.io/models/en-starters#en_trf_robertabase_lg".
        """
        super(DataSet, self).__init__()
        self.cfgs = cfgs

        if cfgs.pretrained_emb['name'] is not None:
            self.pretrained_model = cfgs.pretrained_emb['name']

        self.token2index, self.pretrained_emb = self._load_tp_file()
        self.token_size = self.token2index.__len__()
        self.dataset = self._load_dataset(split)
        self.data_size = self.__len__()

        print('{%3s %5s split size} -> %4d' % (self.cfgs.dataset_use, split, self.data_size))

    def __getitem__(self, index):
        que_iter = self.dataset[0][index]
        opt_iter = self.dataset[1][index]
        cp_ix_iter = self.dataset[2][index]
        ans_iter = self.dataset[3][index]

        return que_iter, opt_iter, cp_ix_iter, ans_iter

    def __len__(self):
        return self.dataset[0].shape[0]

    # load file list
    def _get_list_of_dirs(self, dir_path):
        dirlist = [name for name in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, name))]
        dirlist.sort()
        return dirlist

    # get the list of diagram questions
    def _get_ndq_list(self, ques_path):
        que_list = [name for name in os.listdir(ques_path) if
                    os.path.isdir(os.path.join(ques_path, name)) and name.startswith('NDQ')]
        que_list.sort()
        return que_list

    # get a specific 'png' diagram
    def _get_dia(self, dia_path):
        for name in os.listdir(dia_path):
            if name.endswith('png'):
                return name

    def _load_dataset(self, split):
        proc_data_path = os.path.join(self.cfgs.dataset['data_path'], 'tqa', split,
                                      'processed_data/text_question_sep_files/')
        lesson_list = self._get_list_of_dirs(proc_data_path)
        que_iter = []
        opt_iter = []
        cp_iter = []
        ans_iter = []

        for lesson in lesson_list:
            lesson_path = os.path.join(proc_data_path, lesson)
            ndq_list = self._get_ndq_list(lesson_path)

            for ndq in ndq_list:
                ndq_path = os.path.join(lesson_path, ndq)
                que, opt = self._proc_que_opt(ndq_path)
                cp_per_que = self._proc_cp(ndq_path)
                ans_of_que = self._proc_ans(ndq_path)

                que_iter.append(que)
                opt_iter.append(opt)
                cp_iter.append(cp_per_que)
                ans_iter.append(ans_of_que)

        return torch.from_numpy(np.array(que_iter)), \
               torch.from_numpy(np.array(opt_iter)), \
               torch.from_numpy(np.array(cp_iter)), \
               torch.from_numpy(np.array(ans_iter))

    # load toke2index and pretrained_emb file
    def _load_tp_file(self):
        file_path_prefix = self.cfgs.dataset['data_path']
        token2index_f = os.path.join(file_path_prefix, self.cfgs.dataset_use + '_vocab.json')
        pretrained_emb_f = os.path.join(file_path_prefix, self.cfgs.dataset_use + '_pretrained_emb.npy')

        if os.path.exists(token2index_f):
            with open(token2index_f, 'r') as f:
                token2index = json.load(f)

            pretrained_emb = np.load(pretrained_emb_f)
        else:
            token2index, pretrained_emb = self._token_to_index()
        return token2index, pretrained_emb

    # process strings with regularization expression
    def _proc_strings(self, strings):
        str_list = re.sub(
            r"([.,'!?\"()*#:;])",
            '',
            strings.lower()
        ).replace('-', ' ').replace('/', ' ').replace('_', '').split()
        return str_list

    # process a question with its corresponding options.
    def _proc_que_opt(self, ndq_path):
        opt_ix = np.zeros((self.cfgs.max_ndq_ans, self.cfgs.max_opt_token), np.int64)
        que_ix = np.zeros(self.cfgs.max_que_token, np.int64)
        opt_name = 'a'

        with open(os.path.join(ndq_path, 'Question.txt'), 'r') as f:
            que = f.read()
        que = self._proc_strings(que)
        for index, token in enumerate(que):
            if token in self.token2index:
                que_ix[index] = self.token2index[token]
            else:
                que_ix[index] = self.token2index['UNK']

            if index + 1 == self.cfgs.max_que_token:
                break

        counter = 0
        while os.path.exists(os.path.join(ndq_path, opt_name) + '.txt'):
            with open(os.path.join(ndq_path, opt_name) + '.txt', 'r') as f:
                opt = f.read()
            opt = self._proc_strings(opt)

            for index, token in enumerate(opt):
                if token in self.token2index:
                    opt_ix[counter][index] = self.token2index[token]
                else:
                    opt_ix[counter][index] = self.token2index['UNK']

                if index + 1 == self.cfgs.max_opt_token:
                    break
            counter += 1
            opt_name = chr(ord(opt_name) + 1)
        return que_ix, opt_ix

    # process the closest paragraph
    def _proc_cp(self, ndq_path):
        cp_index = np.zeros((self.cfgs.max_sent, self.cfgs.max_sent_token), np.int64)  # [max_sent, max_sent_token]
        with open(os.path.join(ndq_path, 'closest_sent.txt'), 'r') as f:
            cp = f.readline()  # read the first paragraph from this file.

        par = nlp(cp)
        # use 'en_core_web_sm' to cut sentences within the closest sentence
        for i, sent in enumerate(par.sents):
            sent_token = self._proc_strings(sent.lower_)
            for index, token in enumerate(sent_token):
                if token in self.token2index:
                    cp_index[i][index] = self.token2index[token]
                else:
                    cp_index[i][index] = self.token2index['UNK']

                if index + 1 == self.cfgs.max_sent_token:
                    break
            if i + 1 == self.cfgs.max_sent:
                break
        return cp_index

    # process the correct answer
    def _proc_ans(self, ndq_path):
        ans_index = np.zeros(self.cfgs.max_ndq_ans, np.int64)

        with open(os.path.join(ndq_path, 'correct_answer.txt'), 'r') as f:
            ans = f.readlines()[0].lower().replace('\n', '')
        ans_ascii = ord(ans)
        ans_index[ans_ascii - 97] = 1
        return ans_index

    def _token_to_index(self):
        """
        :return: token_to_index, pretrained word embeddings
        """
        print('Processing dataset to get vocabulary-to-index file ...')
        token_to_ix = {
            'PAD': 0,
            'UNK': 1,
            'CLS': 2
        }
        pretrained_emb = []
        spacy_tool = None
        if self.pretrained_model:
            pretrained_model = importlib.import_module(self.pretrained_model)
            spacy_tool = pretrained_model.load()
            pretrained_emb.append(spacy_tool('PAD').vector)
            pretrained_emb.append(spacy_tool('UNK').vector)
            pretrained_emb.append(spacy_tool('CLS').vector)

        # list train val and test split
        path_pref = self.cfgs.dataset['data_path'] + '/tqa/'

        split_list = os.listdir(path_pref)
        for split in split_list:
            lesson_path = path_pref + split + '/processed_data/text_question_sep_files/'
            lesson_list = os.listdir(lesson_path)

            for lesson in tqdm(lesson_list):
                # process topics.txt
                with open(os.path.join(lesson_path, lesson, 'topics.txt'), 'r') as f:
                    topics = f.readlines()
                    for topic in topics:
                        topic = self._proc_strings(topic)

                        for token in topic:
                            if token not in token_to_ix:
                                token_to_ix[token] = len(token_to_ix)
                                if self.pretrained_model:
                                    pretrained_emb.append(spacy_tool(token).vector)

                # process questions including diagram-questions and non-diagram questions
                questions = self._get_ndq_list(os.path.join(lesson_path, lesson))
                for que in questions:
                    que_path = os.path.join(lesson_path, lesson, que)
                    with open(os.path.join(que_path, 'Question.txt'), 'r') as f:
                        que = f.read()
                        que = self._proc_strings(que)

                        for token in que:
                            if token not in token_to_ix:
                                token_to_ix[token] = len(token_to_ix)
                                if self.pretrained_model:
                                    pretrained_emb.append(spacy_tool(token).vector)

                    opt_flag = 'a'
                    while os.path.exists(os.path.join(que_path, opt_flag + '.txt')):
                        with open(os.path.join(que_path, opt_flag + '.txt'), 'r') as f:
                            opt = f.read()
                            opt = self._proc_strings(opt)

                            for token in opt:
                                if token not in token_to_ix:
                                    token_to_ix[token] = len(token_to_ix)
                                    if self.pretrained_model:
                                        pretrained_emb.append(spacy_tool(token).vector)
                        opt_flag = chr(ord(opt_flag) + 1)

        with open(os.path.join(self.cfgs.dataset['data_path'], self.cfgs.dataset_use + '_vocab.json'), 'w') as f:
            json.dump(token_to_ix, f)
        pretrained_emb = np.array(pretrained_emb)
        np.save(os.path.join(self.cfgs.dataset['data_path'], self.cfgs.dataset_use + '_pretrained_emb'), pretrained_emb)
        return token_to_ix, pretrained_emb
