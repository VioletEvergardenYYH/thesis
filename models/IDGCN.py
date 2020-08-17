# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import DynamicLSTM
from layers import GraphAttentionLayer
from layers import GraphConvolution

class GAT(nn.Module):
    """
    nfeat: input dim
    nhid: output dim
    nclass: GAT最后一层输出纬度
    """
    def __init__(self, nfeat, nhid, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        #self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        #x = F.dropout(x, self.dropout, training=self.training)
        #x = F.elu(self.out_att(x, adj))
        #return F.log_softmax(x, dim=1)
        return x



class IDGCN(nn.Module):
    def __init__(self,  opt):
        super(IDGCN, self).__init__()
        self.opt = opt
        #self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.text_lstm = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.gc1 = GraphConvolution(2*opt.hidden_dim, 2*opt.hidden_dim) #BLSTM所以乘二
        self.gc2 = GraphConvolution(2*opt.hidden_dim, 2*opt.hidden_dim)
        self.gat1 = GAT(2*opt.hidden_dim, 2*opt.hidden_dim, 0.1, 0.2, 3)
        self.fc = nn.Linear(512, opt.polarities_dim)
        self.fc1 = nn.Linear(2*opt.hidden_dim, 512)
        self.nn_drop = nn.Dropout(0.1)
        self.text_embed_dropout = nn.Dropout(0.1)

    def position_weight(self, x, contra_pos, text_len):
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        text_len = text_len.cpu().numpy()

        weight = [[] for i in range(batch_size)]  #每一个样本有一个weight
        for i in range(batch_size):
            #第i个样本的aspect word位于aspect_double_idx[i,0]到aspect_double_idx[i,1]之间
            #这步是论文中GCN第一层的mask掉aspect word及变换权重
            for j in range(text_len[i]):
                if contra_pos[i].count(j):
                    weight[i].append(0)
                else:
                    weight[i].append(1)
            for j in range(text_len[i], seq_len):
                weight[i].append(0)
        weight = torch.tensor(weight).unsqueeze(2).to(self.opt.device)  #batch_size*seqlen*1
        #unsqueeze在指定位置加入维度1，扩充数据维度
        return weight*x

    def mask(self, x, contra_pos):   #针对一个batch size
        batch_size, seq_len = x.shape[0], x.shape[1]
        #contra_pos = contra_pos.cpu().numpy()
        mask = [[] for i in range(batch_size)]
        for i in range(batch_size):
            for j in range(seq_len):
                if contra_pos[i].count(j):
                    mask[i].append(1)
                else:
                    mask[i].append(0)
        mask = torch.tensor(mask).unsqueeze(2).float().to(self.opt.device)
        return mask*x

    def forward(self, inputs):   #每次传入一个batch的数据
        text_bert,batch_text_len, contra_pos, adj = inputs   #text_bert 32*max_len*768 batch_text_len 32*1
        #text_len = torch.sum(text_elmo != 0, dim=-1)
        #text = self.embed(text_elmo)
        text = self.text_embed_dropout(text_bert)
        text_out, (_, _) = self.text_lstm(text, batch_text_len)
        #print('text_out:', text_out.size())              batch_size, len, 2*1024
        #print('batch_text_len:', batch_text_len.size())  batch_size,1
        #print('adj:', adj.size())                        batch_size, len, len
        # x = F.relu(self.gc1(text_out, adj))
        # x = F.relu(self.gc2(x, adj))

        x = F.relu(self.gat1(text_out, adj))
        #print('x before m',x.size())  #batch_size, len, 2048
        x = self.mask(x, contra_pos)
        #print('after m',x.size())   batch_size, len, 2048
        alpha_mat = torch.matmul(x, text_out.transpose(1, 2))   #text_out和x做内积，len*len的内积矩阵,只有contra_pos位置非0，attention结果
        #print('alpha_mat', alpha_mat.size())  batch_size, len, len
        alpha = F.softmax(alpha_mat.sum(1, keepdim=True), dim=2)#32*1*len
        #print('alpha_mat after softmax', alpha.size())   batch_size, 1, len
        x = torch.matmul(alpha, text_out).squeeze(1) # batch_size x 2*hidden_dim   32*1024
        #print('final x', x.size())
        x = self.fc1(x)
        x = self.nn_drop(x)
        output = self.fc(x)
        return output