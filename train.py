import os
import math
import argparse
import random
import numpy
import pdb
import torch
import torch.nn as nn
from bucket_iterator import BucketIterator
from sklearn import metrics
from data_utils import IDDatesetReader
from models import IDGCN


class Instructor:
    def __init__(self, opt):
        self.opt = opt

        id_dataset = IDDatesetReader(dataset=opt.dataset, embed_dim=opt.embed_dim)
        #id_dataset.train_data是一个对象，可以使用[]下标和len
        # print(len(id_dataset.train_data))
        # for i in range(10):
        #     print(id_dataset.train_data[i])

        self.train_data_loader = BucketIterator(data=id_dataset.train_data, batch_size=opt.batch_size, shuffle=True, label = 'train')
        self.test_data_loader = BucketIterator(data=id_dataset.test_data, batch_size=opt.batch_size, shuffle=False, label = 'test')

        self.model = opt.model_class(opt).to(opt.device)
        self._print_args()
        self.global_f1 = 0

        if torch.cuda.is_available():
            print('cuda memory allocated:', torch.cuda.memory_allocated(device=opt.device.index))

    def _print_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape)).item()
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        print('n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        print('> training arguments:')
        for arg in vars(self.opt):
            print('>>> {0}: {1}'.format(arg, getattr(self.opt, arg)))

    def _reset_params(self):
        for p in self.model.parameters():
            if p.requires_grad:
                if len(p.shape) > 1:
                    self.opt.initializer(p)
                else:
                    stdv = 1. / math.sqrt(p.shape[0])
                    torch.nn.init.uniform_(p, a=-stdv, b=stdv)  # 一维可训练参数采用均匀分布

    def _train(self, criterion, optimizer):
        max_test_acc = 0
        max_test_f1 = 0
        max_test_p = 0
        max_test_r = 0
        global_step = 0
        continue_not_increase = 0
        for epoch in range(self.opt.num_epoch):
            print('>' * 100)
            print('epoch: ', epoch)
            n_correct, n_total = 0, 0
            increase_flag = False
            for i_batch, sample_batched in enumerate(self.train_data_loader):  # batch num和all_data字典
                global_step += 1

                # switch model to training mode, clear gradient accumulators
                self.model.train()
                optimizer.zero_grad()
                #print(type(sample_batched['text_elmo']))
                inputs = [sample_batched[col].to(self.opt.device) if col != 'contra_pos' and col != 'words' and col != 'id'  \
                          else sample_batched[col] for col in self.opt.inputs_cols]
                
                #列表，包含'text_bert', 'batch_text_len','contra_pos',  'dependency_graph'
                targets = sample_batched['polarity'].to(self.opt.device)  # 0，1，2

                outputs = self.model(inputs)  # softmax分数
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                if global_step % self.opt.log_step == 0:  # 每五个batch看一下训练效果

                    n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                    n_total += len(outputs)
                    train_acc = n_correct / n_total

                    test_acc, test_f1, test_p, test_r = self._evaluate_acc_f1()
                    if test_acc > max_test_acc:
                        max_test_acc = test_acc
                    if test_f1 > max_test_f1:
                        increase_flag = True
                        max_test_f1 = test_f1
                        max_test_r = test_r
                        max_test_p = test_p
                        if self.opt.save and test_f1 > self.global_f1:
                            self.global_f1 = test_f1
                            torch.save(self.model.state_dict(),
                                       'state_dict/' + 'idgcn' + '.pkl')
                            print('>>> best model saved.')
                    print('loss: {:.4f}, acc: {:.4f}, test_acc: {:.4f}, test_f1: {:.4f}, test_p: {:.4f}, test_r: {:.4f}'.format(loss.item(), train_acc,
                                                                                                test_acc, test_f1, test_p, test_r))
            if increase_flag == False:
                continue_not_increase += 1
                if continue_not_increase >= 3:
                    print('early stop.')
                    break
            else:
                continue_not_increase = 0
        return max_test_acc, max_test_f1, max_test_p, max_test_r

    def _evaluate_acc_f1(self):
        # switch model to evaluation mode
        if self.opt.load_model:
            self.model.load_state_dict(torch.load('state_dict/idgcn.pkl'))
        self.model.eval()
        n_test_correct, n_test_total = 0, 0
        t_targets_all, t_outputs_all = None, None
        with torch.no_grad():
            for t_batch, t_sample_batched in enumerate(self.test_data_loader):
                t_inputs = []
                for col in self.opt.inputs_cols:
                    if (col !='contra_pos' and col != 'words' and col != 'id'):
                        t_inputs.append(t_sample_batched[col].to(opt.device))
                    else:
                        t_inputs.append(t_sample_batched[col])
                t_targets = t_sample_batched['polarity'].to(opt.device)
                if self.opt.load_model:
                    print(t_targets)
                t_outputs = self.model(t_inputs)

                n_test_correct += (torch.argmax(t_outputs, -1) == t_targets).sum().item()
                n_test_total += len(t_outputs)

                if t_targets_all is None:
                    t_targets_all = t_targets
                    t_outputs_all = t_outputs
                else:
                    t_targets_all = torch.cat((t_targets_all, t_targets), dim=0)
                    t_outputs_all = torch.cat((t_outputs_all, t_outputs), dim=0)
            

        test_acc = n_test_correct / n_test_total
        f1 = metrics.f1_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(),
                              average='binary')
        p = metrics.precision_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(),
                              average='binary')
        r = metrics.recall_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(),
                              average='binary')
        return test_acc, f1, p, r

    def run(self, repeats=3):
        # Loss and Optimizer
        criterion = nn.CrossEntropyLoss()
        _params = filter(lambda p: p.requires_grad, self.model.parameters())  # 得到所有可训练参数
        optimizer = self.opt.optimizer(_params, lr=self.opt.learning_rate, weight_decay=self.opt.l2reg)

        if not os.path.exists('log/'):
            os.mkdir('log/')

        f_out = open('log/' + self.opt.model_name + '_' + self.opt.dataset + '_val.txt', 'w', encoding='utf-8')

        max_test_acc_avg = 0
        max_test_f1_avg = 0
        max_test_p_avg = 0
        max_test_r_avg = 0
        for i in range(repeats):
            print('repeat: ', (i + 1))
            f_out.write('repeat: ' + str(i + 1))
            self._reset_params()
            max_test_acc, max_test_f1, max_test_p, max_test_r = self._train(criterion, optimizer)
            print('max_test_acc: {0}     max_test_f1: {1}     max_test_p: {2}     max_test_r: {3}'.format(max_test_acc, max_test_f1, max_test_p, max_test_r))
            f_out.write('max_test_acc: {0}     max_test_f1: {1}     max_test_p: {2}     max_test_r: {3}'.format(max_test_acc, max_test_f1, max_test_p, max_test_r))
            max_test_acc_avg += max_test_acc
            max_test_f1_avg += max_test_f1
            max_test_p_avg += max_test_p
            max_test_r_avg += max_test_r
            print('#' * 100)
        print("max_test_acc_avg:", max_test_acc_avg / repeats)
        print("max_test_f1_avg:", max_test_f1_avg / repeats)
        print("max_test_p_avg:", max_test_p_avg / repeats)
        print("max_test_r_avg:", max_test_r_avg / repeats)

        f_out.close()


if __name__ == '__main__':
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='idgcn', type=str)
    parser.add_argument('--dataset', default='twitter', type=str, help='twitter, rest14, lap14, rest15, rest16')
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    parser.add_argument('--learning_rate', default=0.0001, type=float)
    parser.add_argument('--l2reg', default=0.00001, type=float)
    parser.add_argument('--num_epoch', default=100, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--log_step', default=1, type=int)   #五次没有提升停止训练
    parser.add_argument('--embed_dim', default=768, type=int)
    parser.add_argument('--hidden_dim', default=1024, type=int)
    parser.add_argument('--polarities_dim', default=2, type=int)
    parser.add_argument('--save', default=False, type=bool)
    parser.add_argument('--seed', default=776, type=int)
    parser.add_argument('--device', default=None, type=str)
    parser.add_argument('--use_gcn', default=False, type=bool)
    parser.add_argument('--rand_mask', default=False, type=bool)
    parser.add_argument('--only_bert', default=False, type=bool)
    parser.add_argument('--load_model', default=False, type=bool)
    opt = parser.parse_args()

    model_classes = {

        'idgcn': IDGCN,

    }
    input_colses = {

        'idgcn': ['text_bert', 'batch_text_len','contra_pos',  'dependency_graph', 'words', 'id'],

    }
    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal,
        'orthogonal_': torch.nn.init.orthogonal_,
    }
    optimizers = {
        'adadelta': torch.optim.Adadelta,  # default lr=1.0
        'adagrad': torch.optim.Adagrad,  # default lr=0.01
        'adam': torch.optim.Adam,  # default lr=0.001
        'adamax': torch.optim.Adamax,  # default lr=0.002
        'asgd': torch.optim.ASGD,  # default lr=0.01
        'rmsprop': torch.optim.RMSprop,  # default lr=0.01
        'sgd': torch.optim.SGD,
    }
    opt.model_class = model_classes[opt.model_name]
    opt.inputs_cols = input_colses[opt.model_name]
    opt.initializer = initializers[opt.initializer]
    opt.optimizer = optimizers[opt.optimizer]
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
        if opt.device is None else torch.device(opt.device)

    if opt.seed is not None:
        random.seed(opt.seed)
        numpy.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    ins = Instructor(opt)
    if opt.load_model:
        ins._evaluate_acc_f1()
    else:
        ins.run()
    # id_dataset = IDDatesetReader()
    # print(id_dataset.train_data.data[0]['text_indices'])