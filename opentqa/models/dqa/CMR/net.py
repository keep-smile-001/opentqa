import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.tools import make_mask, get_span_feat, get_ix
from opentqa.models.dqa.CMR.layer import BertEncoder,SimCLR,BertLayerNorm,FlattenAtt,GeLU
from torch.nn.utils.weight_norm import weight_norm
import time
#torch.backends.cudnn.enabled = False
class Net(nn.Module):
    def __init__(self, cfgs, pretrained_emb, token_size):
        super(Net, self).__init__()
        self.cfgs = cfgs

        self.embedding = nn.Embedding(
            token_size,
            cfgs.word_emb_size
        )

        if cfgs.pretrained_emb['name']:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_emb))

        # use simclr to encode features of diagrams
        self.simclr = SimCLR(cfgs)

        # use gru to encode languages
        self.encode_lang = nn.GRU(
            input_size=cfgs.word_emb_size,
            hidden_size=cfgs.hidden_size,
            batch_first=True,
        )
        self.flat = FlattenAtt(cfgs)
        self.bert_encoder = BertEncoder(
            cfgs,
        )
        self.hid_dim = hid_dim = self.bert_encoder.dim  # 768
        self.top_k_value = 10
        self.logit_fc1 = nn.Sequential(
            # nn.Linear(hid_dim * 2, hid_dim * 2), ## original:  all 2,  chen: all 4
            nn.Linear(hid_dim * 1, hid_dim * 1),
            GeLU(),
            BertLayerNorm(hid_dim * 1, eps=1e-12),
            # nn.Linear(hid_dim * 1, 2)
        )
        # self.logit_fc1.apply(self.bert_encoder.model.init_bert_weights)

        self.logit_fc2 = nn.Sequential(
            # nn.Linear(hid_dim * 2, hid_dim * 2), ## original:  all 2,  chen: all 4
            nn.Linear(hid_dim * 2, hid_dim * 2),
            GeLU(),
            BertLayerNorm(hid_dim * 2, eps=1e-12),
            nn.Linear(hid_dim * 2, hid_dim)
            # nn.Linear(hid_dim * 1, 2)
        )
        # self.logit_fc2.apply(self.bert_encoder.model.init_bert_weights)

        self.logit_fc3 = nn.Sequential(
            # nn.Linear(hid_dim * 2, hid_dim * 2), ## original:  all 2,  chen: all 4
            nn.Linear(hid_dim * 1, hid_dim * 1),
            GeLU(),
            BertLayerNorm(hid_dim * 1, eps=1e-12),
            # nn.Linear(hid_dim * 2, hid_dim)
            # nn.Linear(hid_dim * 1, 2)
        )
        # self.logit_fc3.apply(self.bert_encoder.model.init_bert_weights)

        self.logit_fc4 = nn.Sequential(
            # nn.Linear(hid_dim * 2, hid_dim * 2), ## original:  all 2,  chen: all 4
            nn.Linear(hid_dim, hid_dim * 2),
            GeLU(),
            BertLayerNorm(hid_dim * 2, eps=1e-12),
            nn.Linear(hid_dim * 2, hid_dim)
        )
        # self.logit_fc4.apply(self.bert_encoder.model.init_bert_weights)

        # self.w1 = Variable(torch.rand(1, 20, 20), requires_grad=True).cuda()
        # self.w2 = Variable(torch.rand(1, 36, 36), requires_grad=True).cuda()
        # self.w3 = Variable(torch.rand(1, 20, 190), requires_grad=True).cuda()
        # self.w4 = Variable(torch.rand(1, 630, 36), requires_grad=True).cuda()

        self.lang_conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=0)
        self.lang_pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.lang_conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0)
        self.lang_pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.lang_fc1 = nn.Linear(32 * 3 * 7, hid_dim)

        self.img_conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=0)
        self.img_pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.img_conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0)
        self.img_pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.img_fc1 = nn.Linear(32 * 7 * 7, hid_dim)

        self.cross_conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=0)
        self.cross_pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # self.cross_conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0)
        # self.cross_pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.cross_fc1 = nn.Linear(32 * 2 * 2, hid_dim)

        self.rel_conv1 = nn.Conv2d(1, 32, kernel_size=1, stride=1, padding=0)
        self.rel_pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # self.rel_conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0)
        # self.rel_pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.rel_fc1 = nn.Linear(32 * 5 * 1, hid_dim)

        #         self.final_classifier = nn.Linear(hid_dim*4, 2)
        self.final_classifier = nn.Sequential(
            # nn.Linear(hid_dim * 2, hid_dim * 2), ## original:  all 2,  chen: all 4
            nn.Linear(hid_dim * 2+1024, hid_dim * 4),
            GeLU(),
            BertLayerNorm(hid_dim * 4, eps=1e-12),
            # nn.Linear(hid_dim * 4, 2)
            # nn.ReLU(),
            nn.Linear(hid_dim * 4, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim // 4),
            nn.ReLU(),
            nn.Linear(hid_dim // 4, 1)
        )

        self.classifer = nn.Sequential(
            weight_norm(nn.Linear(hid_dim * 2+1024, cfgs.flat_out_size), dim=None),
            nn.ReLU(),
            nn.Dropout(cfgs.classifer_dropout_r, inplace=True),
            weight_norm(nn.Linear(cfgs.flat_out_size, 1), dim=None)
        )
        ## for relationship
        self.lang_2_to_1 = nn.Sequential(
            nn.Linear(hid_dim * 2, hid_dim * 2),
            nn.ReLU(),
            nn.Linear(hid_dim * 2, hid_dim),
        )
        self.img_2_to_1 = nn.Sequential(
            nn.Linear(hid_dim * 2, hid_dim * 2),
            nn.ReLU(),
            nn.Linear(hid_dim * 2, hid_dim),
        )
        self.lang_relation = nn.Sequential(
            nn.Linear(hid_dim, hid_dim // 2),
            nn.ReLU(),
            nn.Linear(hid_dim // 2, hid_dim // 4),
            nn.ReLU(),
            nn.Linear(hid_dim // 4, 1)
        )
        self.img_relation = nn.Sequential(
            nn.Linear(hid_dim, hid_dim // 2),
            nn.ReLU(),
            nn.Linear(hid_dim // 2, hid_dim // 4),
            nn.ReLU(),
            nn.Linear(hid_dim // 4, 1)
        )






    def forward(self, que_ix, opt_ix, dia, ins_dia, cp_ix):
        """
        :param que_ix: the index of questions
        :param opt_ix: the index of options
        :param dia: the diagram corresponding the above question
        :param ins_dia: the instructional diagram corresponding to the lesson that contains the above question
        :param cp_ix: the closest paragraph that is extracted by TF-IDF method
        """

        batch_size = que_ix.shape[0]
        que_mask = make_mask(que_ix.unsqueeze(2))
        que_feat = self.embedding(que_ix)
        que_feat, _ = self.encode_lang(que_feat)
        que_feat = que_feat.reshape(batch_size, -1, self.hid_dim)
        ##print(que_feat.size())
        opt_ix = opt_ix.reshape(-1, self.cfgs.max_opt_token)
        opt_mask = make_mask(opt_ix.unsqueeze(2))

        opt_feat = self.embedding(opt_ix)
        opt_feat, _ = self.encode_lang(opt_feat)
        opt_feat = self.flat(opt_feat, opt_mask)
        opt_feat = opt_feat.reshape(batch_size, self.cfgs.max_ans, -1)

        dia_feat = self.simclr(dia)
        dia_feat = dia_feat.reshape(batch_size, -1)

        batch_size, img_num, obj_num, feat_size =  dia_feat.size()[0],1 ,1 , dia_feat.size()[1]
        ##print(dia_feat.size())
        assert img_num == 1 and obj_num == 1 and feat_size == 2048
        dia_feat = dia_feat.reshape(batch_size , obj_num, feat_size)

        output_lang, output_img, output_cross = self.bert_encoder(sents=que_feat,feats=dia_feat)
        # output_cross = output_cross.view(-1, self.hid_dim*2) ## original
        output_cross = output_cross.view(-1, self.hid_dim)
        ##print(output_lang.size(), output_img.size(), output_cross.size())

        #### new experiment for relationship
        relate_lang_stack_1 = output_lang.view(output_lang.size()[0], 1, output_lang.size()[1], output_lang.size()[2])
        relate_lang_stack_2 = output_lang.view(output_lang.size()[0], output_lang.size()[1], 1, output_lang.size()[2])
        # relate_lang_stack = relate_lang_stack_1 + relate_lang_stack_2 ## [64, 20, 20, 768]
        relate_lang_stack_1 = relate_lang_stack_1.repeat(1, output_lang.size()[1], 1,
                                                         1)  ## [64, 20, 20, 768] second dim repeat 10 times, others not change
        relate_lang_stack_2 = relate_lang_stack_2.repeat(1, 1, output_lang.size()[1],
                                                         1)  ## [64, 20, 20, 768] third dim repeat 10 times, others not change
        relate_lang_stack = torch.cat((relate_lang_stack_1, relate_lang_stack_2), 3)  ## [64, 20, 20, 768*2]

        relate_lang_stack = relate_lang_stack.view(-1, output_lang.size()[2] * 2)
        relate_lang_stack = self.lang_2_to_1(relate_lang_stack)
        relate_lang_stack = relate_lang_stack.view(output_lang.size()[0], output_lang.size()[1], output_lang.size()[1],
                                                   output_lang.size()[2])

        relate_img_stack_1 = output_img.view(output_img.size()[0], 1, output_img.size()[1], output_img.size()[2])#[16,1,1,768]
        relate_img_stack_2 = output_img.view(output_img.size()[0], output_img.size()[1], 1, output_img.size()[2])#[16,1,1,768]
        ##print("r_i_s",relate_img_stack_1.size(),relate_img_stack_2.size())
        relate_img_stack = relate_img_stack_1 + relate_img_stack_2  ## [16, 1, 1, 768]
        # relate_img_stack_1 = relate_img_stack_1.repeat(1,output_img.size()[1],1,1)  ## [64, 20, 20, 768] second dim repeat 10 times, others not change
        # relate_img_stack_2 = relate_img_stack_2.repeat(1,1,output_img.size()[1],1)  ## [64, 20, 20, 768] third dim repeat 10 times, others not change
        # relate_img_stack = torch.cat((relate_img_stack_1, relate_img_stack_2), 3)

        # relate_img_stack = relate_img_stack.view(-1, output_lang.size()[2]*2)
        # relate_img_stack = self.lang_2_to_1(relate_img_stack)
        # relate_img_stack = relate_img_stack.view(output_img.size()[0], output_img.size()[1], output_img.size()[1], output_img.size()[2])

        relate_lang_stack = relate_lang_stack.view(relate_lang_stack.size()[0],
                                                   relate_lang_stack.size()[1] * relate_lang_stack.size()[2],
                                                   relate_lang_stack.size()[3])  ## [64, 400, 768] or 768*2
        relate_img_stack = relate_img_stack.view(relate_img_stack.size()[0],
                                                 relate_img_stack.size()[1] * relate_img_stack.size()[2],
                                                 relate_img_stack.size()[3])  ## [64, 1296, 768] or 768*2
        ##print("relate_i_s",relate_img_stack.size())
        ### a beautiful way
        relate_lang_ind = torch.tril_indices(output_lang.size()[1], output_lang.size()[1], -1).cuda(0)
        relate_lang_ind[1] = relate_lang_ind[1] * output_lang.size()[1]
        relate_lang_ind = relate_lang_ind.sum(0)
        relate_lang_stack = relate_lang_stack.index_select(1, relate_lang_ind)  ## [64, 190, 768] or 768*2
        '''
        relate_img_ind = torch.tril_indices(output_img.size()[1], output_img.size()[1], -1).cuda(0)
        #print("rii", relate_img_ind)
        relate_img_ind[1] = relate_img_ind[1] * output_img.size()[1]
        relate_img_ind = relate_img_ind.sum(0)
        #print("rii",relate_img_ind.size())
        relate_img_stack = relate_img_stack.index_select(1, relate_img_ind)  ## [64, 630, 768] or 768*2
        #print("ris", relate_img_stack.size())'''
        ## reshape the relate_lang_stack and relate_img_stack
        tmp_lang_stack = relate_lang_stack.view(-1, self.hid_dim)  # sum
        tmp_img_stack = relate_img_stack.view(-1, self.hid_dim)  # sum
        # tmp_lang_stack = relate_lang_stack.view(-1, self.hid_dim*2)   # cat
        # tmp_img_stack = relate_img_stack.view(-1, self.hid_dim*2)     # cat

        lang_candidate_relat_score = self.lang_relation(tmp_lang_stack)
        img_candidate_relat_score = self.img_relation(tmp_img_stack)

        lang_candidate_relat_score = lang_candidate_relat_score.view(output_lang.size()[0],
                                                                     relate_lang_stack.size()[1])  ##(64, 190)
        img_candidate_relat_score = img_candidate_relat_score.view(output_img.size()[0],
                                                                   relate_img_stack.size()[1])  ## (64,630)
        ##print("icrs",img_candidate_relat_score)
        ##print("-1")
        #time.sleep(10)
        _, topk_lang_index = torch.topk(lang_candidate_relat_score, self.top_k_value, sorted=False)  ##(64, 10)
        #_, topk_img_index = torch.topk(img_candidate_relat_score, self.top_k_value, sorted=False)  ##(64, 10)
        ##print("0")
        #time.sleep(10)
        list_lang_relat = []
        list_img_relat = []
        for i in range(0, output_lang.size()[0]):
            tmp = torch.index_select(relate_lang_stack[i], 0, topk_lang_index[i])  ## [10, 768] or 768*2
            list_lang_relat.append(tmp)
        #print("1")
        #time.sleep(10)
        for i in range(0, output_img.size()[0]):
            tmp = relate_img_stack[i]  ## [10, 768] or 768*2
            list_img_relat.append(tmp)
            #print("tmp",tmp.size())
        #print("2")
        #time.sleep(10)
        #raise AssertionError
        #print("l-i-r", list_img_relat)
        lang_relat = torch.cat(list_lang_relat, 0)  ## [640, 768] or 768*2
        img_relat = torch.cat(list_img_relat, 0)  ## [640, 768] or 768*2
        #print("l-i-r", lang_relat.size(), img_relat.size())
        lang_relat = lang_relat.view(output_lang.size()[0], -1, self.hid_dim)  ## [64, 10, 768] or 768*2
        img_relat = img_relat.view(output_img.size()[0], -1, self.hid_dim)  ## [64, 10, 768] or 768*2
        # lang_relat = lang_relat.view(output_lang.size()[0], -1, self.hid_dim*2) ## [64, 10, 768] or 768*2
        # img_relat = img_relat.view(output_img.size()[0], -1, self.hid_dim*2) ## [64, 10, 768] or 768*2
        #print("l-i-r",lang_relat.size(),img_relat.size())

        relate_cross = torch.einsum(
            'bld,brd->blr',
            F.normalize(lang_relat, p=2, dim=-1),
            F.normalize(img_relat, p=2, dim=-1)
        )
        #print("rc", relate_cross.size())
        relate_cross = relate_cross.view(-1, 1, relate_cross.size()[1], relate_cross.size()[2])
        realte_conv_1 = self.rel_pool1(F.relu(self.rel_conv1(relate_cross)))
        # realte_conv_2 = self.rel_pool2(F.relu(self.rel_conv2(realte_conv_1)))
        ##print("rcsss", realte_conv_1.size())
        relate_fc1 = F.relu(self.rel_fc1(realte_conv_1.view(-1, 32 * 5 * 1)))
        ##print(relate_fc1.size())
        relate_fc1 = relate_fc1.view(-1, self.hid_dim)
        logit4 = self.logit_fc4(relate_fc1)

        #### new experiment for cross modality
        output_cross = output_cross.view(-1, output_cross.size()[1])
       # #print("oc:",output_cross.size())
        cross_tuple = torch.split(output_cross, output_cross.size()[1] // 2, dim=1)
        ##print("ct:",cross_tuple[0].size(),cross_tuple[1].size())
        cross1 = cross_tuple[0].view(output_cross.size()[0], -1, 64)
        cross2 = cross_tuple[1].view(output_cross.size()[0], -1, 64)
        ##print("cross1",cross1.size(),cross2.size())
        ##print("c",output_cross.size())

       # #print(F.normalize(cross1,p=2,dim=-1).size())
        cross_1_2 = torch.einsum(
            'bld,brd->blr',
            F.normalize(cross1, p=2, dim=-1),
            F.normalize(cross2, p=2, dim=-1)
        )
        ##print("oc:", cross_1_2.size())
        cross_1_2 = cross_1_2.view(-1, 1, cross_1_2.size()[1], cross_1_2.size()[2])
        cross_conv_1 = self.cross_pool1(F.relu(self.cross_conv1(cross_1_2)))
        # cross_conv_2 = self.cross_pool2(F.relu(self.cross_conv2(cross_conv_1)))
        '''
        #### new experiment for lang and two images
        cross_img_sen = torch.einsum(
            'bld,brd->blr',
            F.normalize(output_lang, p=2, dim=-1),
            F.normalize(output_img, p=2, dim=-1)
        )

        cross_img_sen = cross_img_sen.view(-1, 1, cross_img_sen.size()[1], cross_img_sen.size()[2])
        entity_conv_1 = self.lang_pool1(F.relu(self.lang_conv1(cross_img_sen)))
        entity_conv_2 = self.lang_pool2(F.relu(self.lang_conv2(entity_conv_1)))

        ### new experiment for two images
        image_2_together = output_img.view(-1, output_img.size()[1], self.hid_dim * 2)
        # #print(image_2_together.size())
        images = torch.split(image_2_together, self.hid_dim // 2, dim=2)
        # #print(images[0].size(), images[1].size())
        
        image1 = images[0]
        image2 = images[1]
        '''
        '''
        cross_img_img = torch.einsum(
            'bld,brd->blr',
            F.normalize(image1, p=2, dim=-1),
            F.normalize(image2, p=2, dim=-1)
        )
        cross_img_img = cross_img_img.view(-1, 1, cross_img_img.size()[1], cross_img_img.size()[2])
        cross_img_conv_1 = self.img_pool1(F.relu(self.img_conv1(cross_img_img)))
        cross_img_conv_2 = self.img_pool2(F.relu(self.img_conv2(cross_img_conv_1)))
        # #print(cross_img_conv_2.size())

        img_fc1 = F.relu(self.img_fc1(cross_img_conv_2.view(-1, 32*7*7)))
        img_fc1 = img_fc1.view(-1, self.hid_dim)
        logit3 = self.logit_fc3(img_fc1)
        '''


        '''
        entity_fc1 = F.relu(self.lang_fc1(entity_conv_2.view(-1, 32 * 3 * 7)))
        entity_fc1 = entity_fc1.view(-1, self.hid_dim * 2)
        logit2 = self.logit_fc2(entity_fc1)
        '''

        ##print('cc1',cross_conv_1.size())
        cross_fc1 = F.relu(self.cross_fc1(cross_conv_1.view(-1, 32 * 2 * 2)))
        cross_fc1 = cross_fc1.view(-1, self.hid_dim)
        logit1 = self.logit_fc1(cross_fc1)
        # #print(logit1.size(),logit2.size(),logit3.size(),logit4.size())

        #cross_logit = torch.cat((logit1, logit2, logit3, logit4), 1)
        cross_logit = torch.cat((logit1, logit4), 1)
       # #print("final:",logit1.size(),logit4.size())
        ##print("all:",cross_logit.size())
        cross_logit = cross_logit.repeat(1, self.cfgs.max_ans).reshape(batch_size, self.cfgs.max_ans, -1)
        #print("after all:", cross_logit.size(),opt_feat.size())
        cross_logit = torch.cat((cross_logit, opt_feat), dim=-1)
        #print("?",cross_logit.size())
        logit = self.final_classifier(cross_logit).squeeze(-1)


        #logit = self.classifer(cross_logit).squeeze(-1)
        #print("logit",logit.size())
        return logit