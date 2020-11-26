# -----------------------------------------------
# !/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time:2020/9/16 16:32
# -----------------------------------------------
import torch, time, datetime, os, operator, wandb
import numpy as np
import torch.utils.data as Data
import torch.nn as nn
from opentqa.datasets.dataset_loader import DatasetLoader
from opentqa.models.model_loader import ModelLoader
from utils.tools import write_log
from utils.optim import adjust_lr, get_optim


class Engine(object):
    def __init__(self, cfgs):
        self.cfgs = cfgs

    def _get_device(self):
        device = 'cuda: {}'.format(self.cfgs.gpu) if torch.cuda.is_available() else 'cpu'
        return device

    # get accuracy on dqa dataset
    def _get_accuracy_on_dqa(self, pred, ans):
        pred = np.array(pred)
        ans = np.array(ans)

        que_sum = pred.size
        acc_sum = (pred == ans).sum()

        acc = acc_sum / float(que_sum)
        contents = 'Overall Accuracy is ' + str(acc) + '\n'
        print(contents + '\n')

        if self.cfgs.run_mode == 'test':
            write_log(self.cfgs, self.cfgs)
            write_log(self.cfgs, '=========================================================' +
                      '\n' + contents)
        else:
            write_log(self.cfgs, contents)
        return acc

        # get accuracy on dqa dataset

    def _get_accuracy_on_ndqa(self, pred, ans, type):
        pred = np.array(pred)
        ans = np.array(ans)

        que_sum = pred.size
        acc_sum = (pred == ans).sum()

        type = np.array(type)
        tf = (type == 2)
        tf_sum = tf.sum()
        mc_sum = que_sum - tf_sum

        acc_tf = ((pred == ans) & tf).sum()
        acc_mc = acc_sum - acc_tf

        acc_all = acc_sum / float(que_sum)
        acc_tf = acc_tf / float(acc_sum)
        acc_mc = acc_mc / float(mc_sum)

        contents = 'Overall Accuracy is {} -> [TF Accuracy is {} || MC Accuracy is {}]'.format(str(acc_all)[0:7],
                                                                                               str(acc_tf)[0:7],
                                                                                               str(acc_mc)[0:7])

        print(contents + '\n')

        if self.cfgs.run_mode == 'test':
            write_log(self.cfgs, self.cfgs)
            write_log(self.cfgs, '=========================================================' +
                      '\n' + contents)
        else:
            write_log(self.cfgs, contents)
        return acc_all

    def ckpt_proc(self, state_dict):
        state_dict_new = {}
        for key in state_dict:
            state_dict_new['module.' + key] = state_dict[key]

        return state_dict_new

    # dynamically load method according to name
    def load_method(self):
        if self.cfgs.run_mode == 'train':
            train_dataset = DatasetLoader(cfgs=self.cfgs, split='train').dataset()
            val_dataset = DatasetLoader(cfgs=self.cfgs, split='test').dataset()

            return operator.methodcaller('train_on_' + self.cfgs.dataset_use, train_dataset, val_dataset)(self)
        else:
            test_dataset = DatasetLoader(cfgs=self.cfgs, split='test').dataset()

            return operator.methodcaller('validate_on_' + self.cfgs.dataset_use, test_dataset)(self)

    # train models on the dqa training dataset
    def train_on_dqa(self, dataset, val_dataset):
        token_size = dataset.token_size
        pretrained_emb = dataset.pretrained_emb
        device = self._get_device()

        net = ModelLoader(self.cfgs).Net(
            self.cfgs,
            pretrained_emb,
            token_size
        )
        # wandb.watch(net)
        net.to(device)
        net.train()

        if self.cfgs.n_gpu > 1:
            net = nn.DataParallel(net, device_ids=self.cfgs.devices)

        data_loader = Data.DataLoader(
            dataset=dataset,
            batch_size=int(self.cfgs.batch_size),
            shuffle=True,
            num_workers=int(self.cfgs.dataset['num_works']),
            drop_last=True
        )

        optimizer = get_optim(self.cfgs, net, dataset.data_size)
        criterion = nn.CrossEntropyLoss(reduction=self.cfgs.reduction)

        write_log(cfgs=self.cfgs, content=self.cfgs)
        loss_sum = 0

        for epoch in range(int(self.cfgs.epoch)):
            content = '=========================================================' + \
                      '\ncurrent time: ' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '\n'
            write_log(cfgs=self.cfgs, content=content)

            start_time = time.time()
            # Learning Rate Decay
            if epoch in self.cfgs.lr_decay_list:
                adjust_lr(optimizer, self.cfgs.lr_decay_r)

            for step, (
                    que_iter,
                    opt_iter,
                    dia_iter,
                    ins_dia_iter,
                    cp_ix_iter,
                    ans_iter
            ) in enumerate(data_loader):
                optimizer.zero_grad()

                que_iter = que_iter.to(self._get_device())
                opt_iter = opt_iter.to(self._get_device())
                dia_iter = dia_iter.to(self._get_device())
                ins_dia_iter = ins_dia_iter.to(self._get_device())
                cp_ix_iter = cp_ix_iter.to(self._get_device())
                ans_iter = ans_iter.to(self._get_device())

                pred = net(
                    que_iter,
                    opt_iter,
                    dia_iter,
                    ins_dia_iter,
                    cp_ix_iter
                )
                _, ans = torch.max(ans_iter, dim=-1)
                _, pred_ix = torch.max(pred, dim=-1)

                loss = criterion(pred, ans)
                loss.backward()
                loss_sum += loss.cpu().data.numpy()

                print("\r[Version: %s][Model: %s][Dataset: %s][Epoch %2d][Step %4d/%4d] Loss: %.4f, Lr: %.2e" % (
                    self.cfgs.version,
                    self.cfgs.model,
                    self.cfgs.dataset_use,
                    epoch + 1,
                    step + 1,
                    int(dataset.data_size / int(self.cfgs.batch_size)),
                    loss / float(self.cfgs.batch_size),
                    optimizer._rate
                ), end='          ')

                # gradient norm clipping
                if self.cfgs.grad_norm_clip > 0:
                    nn.utils.clip_grad_norm_(net.parameters(), self.cfgs.grad_norm_clip)

                optimizer.step()

            end_time = time.time()
            time_consuming = end_time - start_time
            print('Finished in {}s'.format(int(time_consuming)))

            # save checkpoints
            if self.cfgs.n_gpu > 1:
                state = {'state_dict': net.module.state_dict()}
            else:
                state = {'state_dict': net.state_dict()}

            # check ckpt_version path
            if ('ckpt_' + self.cfgs.version) not in os.listdir(self.cfgs.ckpts_path):
                os.mkdir(self.cfgs.ckpts_path + '/ckpt_' + self.cfgs.version)

            torch.save(state, self.cfgs.ckpts_path +
                       '/ckpt_' + self.cfgs.version +
                       '/epoch' + str(epoch + 1) + '.pkl')

            content = 'Epoch: ' + str(epoch + 1) + \
                      ', Train Loss: ' + str(loss_sum / dataset.data_size) + \
                      ', Lr: ' + str(optimizer._rate) + '\n' + \
                      'Time consuming: ' + str(int(time_consuming)) + \
                      ', Speed(s/batch): ' + str(time_consuming / step) + \
                      '\n\n'

            write_log(cfgs=self.cfgs, content=content)
            loss_sum = 0

            self.validate_on_dqa(val_dataset, net.state_dict())

    # validate models on the dqa val_dataset
    def validate_on_dqa(self, dataset, state_dict=None):
        if state_dict is None:
            path = self.cfgs.ckpts_path + \
                   '/ckpt_' + self.cfgs.version + \
                   '/epoch' + self.cfgs.ckpt_epoch + '.pkl'

            print('Loading ckpt from: {}'.format(path))
            state_dict = torch.load(path)['state_dict']
            print('Finish!')

            if self.cfgs.n_gpu > 1:
                state_dict = self.ckpt_proc(state_dict)

        criterion = nn.CrossEntropyLoss(reduction=self.cfgs.reduction)

        if self.cfgs.run_mode == 'test':
            mode = 'Train -> Test'
        else:
            mode = 'Train -> Val'

        with torch.no_grad():
            token_size = dataset.token_size
            pretrained_emb = dataset.pretrained_emb
            device = self._get_device()

            net = ModelLoader(self.cfgs).Net(
                self.cfgs,
                pretrained_emb,
                token_size
            )
            net.to(device)
            net.eval()

            if self.cfgs.n_gpu > 1:
                net = nn.DataParallel(net, device_ids=self.cfgs.devices)
            net.load_state_dict(state_dict)

            data_loader = Data.DataLoader(
                dataset=dataset,
                batch_size=int(self.cfgs.batch_size),
                shuffle=False,
                num_workers=int(self.cfgs.batch_size),
                drop_last=True
            )

            ans_list = []  # the list of the ground truth
            pred_list = []  # the list of the prediction
            loss_sum = 0

            for step, (
                    que_iter,
                    opt_iter,
                    dia_iter,
                    ins_dia_iter,
                    cp_ix_iter,
                    ans_iter
            ) in enumerate(data_loader):
                print('\rEvaluation: [%s][step %4d/%4d]' % (
                    mode,
                    step + 1,
                    int(dataset.data_size / int(self.cfgs.batch_size))
                ), end='    ')

                que_iter = que_iter.to(self._get_device())
                opt_iter = opt_iter.to(self._get_device())
                dia_iter = dia_iter.to(self._get_device())
                ins_dia_iter = ins_dia_iter.to(self._get_device())
                cp_ix_iter = cp_ix_iter.to(self._get_device())
                ans_iter = ans_iter.to(self._get_device())

                pred = net(
                    que_iter,
                    opt_iter,
                    dia_iter,
                    ins_dia_iter,
                    cp_ix_iter
                )

                _, ans = torch.max(ans_iter, dim=-1)
                loss = criterion(pred.squeeze(-1), ans)
                loss_sum += loss.cpu().data.numpy()

                pred_np = pred.cpu().data.squeeze(-1).numpy()
                pred_argmax = np.argmax(pred_np, axis=1)
                pred_list.append(pred_argmax)

                ans_np = ans_iter.cpu().data.numpy()
                ans_argmax = np.argmax(ans_np, axis=1)
                ans_list.append(ans_argmax)

            val_loss = loss_sum / float(dataset.data_size)
            contents = 'Validation Loss: ' + str(val_loss) + '\n'
            write_log(self.cfgs, contents)
            # torch.save(net.state_dict(), os.path.join(wandb.run.dir, 'model.pt'))
            acc = self._get_accuracy_on_dqa(pred_list, ans_list)
            metrics = {'acc': acc, 'val_loss': val_loss}
            # wandb.log(metrics)

    # train models on the ndqa training dataset
    def train_on_ndqa(self, dataset, val_dataset):
        token_size = dataset.token_size
        pretrained_emb = dataset.pretrained_emb
        device = self._get_device()

        net = ModelLoader(self.cfgs).Net(
            self.cfgs,
            pretrained_emb,
            token_size
        )
        # wandb.watch(net)
        net.to(device)
        net.train()

        if self.cfgs.n_gpu > 1:
            net = nn.DataParallel(net, device_ids=self.cfgs.devices)

        data_loader = Data.DataLoader(
            dataset=dataset,
            batch_size=int(self.cfgs.batch_size),
            shuffle=True,
            num_workers=int(self.cfgs.dataset['num_works']),
            drop_last=True
        )

        optimizer = get_optim(self.cfgs, net, dataset.data_size)
        criterion = nn.CrossEntropyLoss(reduction=self.cfgs.reduction)

        write_log(cfgs=self.cfgs, content=self.cfgs)
        loss_sum = 0

        for epoch in range(int(self.cfgs.epoch)):
            content = '=========================================================' + \
                      '\ncurrent time: ' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '\n'
            write_log(cfgs=self.cfgs, content=content)

            start_time = time.time()
            # Learning Rate Decay
            if epoch in self.cfgs.lr_decay_list:
                adjust_lr(optimizer, self.cfgs.lr_decay_r)

            for step, (
                    que_iter,
                    opt_iter,
                    cp_ix_iter,
                    ans_iter
            ) in enumerate(data_loader):
                optimizer.zero_grad()

                que_iter = que_iter.to(self._get_device())
                opt_iter = opt_iter.to(self._get_device())
                cp_ix_iter = cp_ix_iter.to(self._get_device())
                ans_iter = ans_iter.to(self._get_device())

                pred, opt_num = net(
                    que_iter,
                    opt_iter,
                    cp_ix_iter
                )
                _, ans = torch.max(ans_iter, dim=-1)
                _, pred_ix = torch.max(pred, dim=-1)

                loss = criterion(pred, ans)
                loss.backward()
                loss_sum += loss.cpu().data.numpy()

                print("\r[Version: %s][Model: %s][Dataset: %s][Epoch %2d][Step %4d/%4d] Loss: %.4f, Lr: %.2e" % (
                    self.cfgs.version,
                    self.cfgs.model,
                    self.cfgs.dataset_use,
                    epoch + 1,
                    step + 1,
                    int(dataset.data_size / int(self.cfgs.batch_size)),
                    loss / float(self.cfgs.batch_size),
                    optimizer._rate
                ), end='          ')

                # gradient norm clipping
                if self.cfgs.grad_norm_clip > 0:
                    nn.utils.clip_grad_norm_(net.parameters(), self.cfgs.grad_norm_clip)

                optimizer.step()

            end_time = time.time()
            time_consuming = end_time - start_time
            print('Finished in {}s'.format(int(time_consuming)))

            # save checkpoints
            if self.cfgs.n_gpu > 1:
                state = {'state_dict': net.module.state_dict()}
            else:
                state = {'state_dict': net.state_dict()}

            # check ckpt_version path
            if ('ckpt_' + self.cfgs.version) not in os.listdir(self.cfgs.ckpts_path):
                os.mkdir(self.cfgs.ckpts_path + '/ckpt_' + self.cfgs.version)

            torch.save(state, self.cfgs.ckpts_path +
                       '/ckpt_' + self.cfgs.version +
                       '/epoch' + str(epoch + 1) + '.pkl')

            content = 'Epoch: ' + str(epoch + 1) + \
                      ', Train Loss: ' + str(loss_sum / dataset.data_size) + \
                      ', Lr: ' + str(optimizer._rate) + '\n' + \
                      'Time consuming: ' + str(int(time_consuming)) + \
                      ', Speed(s/batch): ' + str(time_consuming / step) + \
                      '\n\n'

            write_log(cfgs=self.cfgs, content=content)
            loss_sum = 0

            self.validate_on_ndqa(val_dataset, net.state_dict())

    # validate models on the ndqa validation dataset
    def validate_on_ndqa(self, dataset, state_dict=None):
        if state_dict is None:
            path = self.cfgs.ckpts_path + \
                   '/ckpt_' + self.cfgs.version + \
                   '/epoch' + self.cfgs.ckpt_epoch + '.pkl'

            print('Loading ckpt from: {}'.format(path))
            state_dict = torch.load(path)['state_dict']
            print('Finish!')

            if self.cfgs.n_gpu > 1:
                state_dict = self.ckpt_proc(state_dict)

        criterion = nn.CrossEntropyLoss(reduction=self.cfgs.reduction)

        if self.cfgs.run_mode == 'test':
            mode = 'Train -> Test'
        else:
            mode = 'Train -> Val'

        with torch.no_grad():
            token_size = dataset.token_size
            pretrained_emb = dataset.pretrained_emb
            device = self._get_device()

            net = ModelLoader(self.cfgs).Net(
                self.cfgs,
                pretrained_emb,
                token_size
            )
            net.to(device)
            net.eval()

            if self.cfgs.n_gpu > 1:
                net = nn.DataParallel(net, device_ids=self.cfgs.devices)
            net.load_state_dict(state_dict)

            data_loader = Data.DataLoader(
                dataset=dataset,
                batch_size=int(self.cfgs.batch_size),
                shuffle=False,
                num_workers=int(self.cfgs.batch_size),
                drop_last=True
            )

            ans_list = []  # the list of the ground truth
            pred_list = []  # the list of the prediction
            type_list = []  # the list of non-diagram questions: T/F or MC
            loss_sum = 0

            for step, (
                    que_iter,
                    opt_iter,
                    cp_ix_iter,
                    ans_iter
            ) in enumerate(data_loader):
                print('\rEvaluation: [%s][step %4d/%4d]' % (
                    mode,
                    step + 1,
                    int(dataset.data_size / int(self.cfgs.batch_size))
                ), end='    ')

                que_iter = que_iter.to(self._get_device())
                opt_iter = opt_iter.to(self._get_device())
                cp_ix_iter = cp_ix_iter.to(self._get_device())
                ans_iter = ans_iter.to(self._get_device())

                pred, opt_num = net(
                    que_iter,
                    opt_iter,
                    cp_ix_iter
                )

                _, ans = torch.max(ans_iter, dim=-1)
                loss = criterion(pred.squeeze(-1), ans)
                loss_sum += loss.cpu().data.numpy()

                pred_np = pred.cpu().data.squeeze(-1).numpy()
                pred_argmax = np.argmax(pred_np, axis=1)
                pred_list.append(pred_argmax)

                ans_np = ans_iter.cpu().data.numpy()
                ans_argmax = np.argmax(ans_np, axis=1)
                ans_list.append(ans_argmax)

                # the number of options 2 means T/F; Otherwise, 4 means MC;
                opt_num = opt_num.cpu().data.numpy()
                type_list.append(opt_num)

            val_loss = loss_sum / float(dataset.data_size)
            contents = 'Validation Loss: ' + str(val_loss) + '\n'
            write_log(self.cfgs, contents)
            # torch.save(net.state_dict(), os.path.join(wandb.run.dir, 'model.pt'))
            acc = self._get_accuracy_on_ndqa(pred_list, ans_list, type_list)
            metrics = {'acc': acc, 'val_loss': val_loss}
            # wandb.log(metrics)
