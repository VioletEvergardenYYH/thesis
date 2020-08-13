# -*- coding: utf-8 -*-
import pdb
import math
import random
import torch
import os
import pickle
import numpy
from allennlp.modules.elmo import Elmo, batch_to_ids
options_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

class BucketIterator(object):
    def __init__(self, data, batch_size, sort_key='text', shuffle=True, sort=True, label = None):
        self.shuffle = shuffle
        self.sort = sort
        self.i = 0
        self.label = label
        self.sort_key = sort_key
        self.batches = self.sort_and_pad(data, batch_size)#返回一个列表，列表每一项是一个batch的数据字典
        self.batch_len = len(self.batches)

    def sort_and_pad(self, data, batch_size):  #返回一个列表，列表每一项是一个batch字典
        #data=id_dataset.train_data
        """
        :param data: 一个列表，每一项是一个样本的数据字典
        :param batch_size: int
        :return: batches 一个列表，每一项是一个batch的数据字典
        """
        num_batch = int(math.ceil(len(data) / batch_size))
        if self.sort:
            sorted_data = sorted(data, key=lambda x: x[self.sort_key].size(0),reverse=True)
        else:
            sorted_data = data
        batches = []

        for i in range(num_batch):
            batches.append(self.pad_data(sorted_data[i*batch_size : (i+1)*batch_size],i))
        print(len(batches))
        # for item in batches:
        #     print(item)
        return batches
    def pad_data(self, batch_data, i):   #返回一个batch的字典
        """

        :param batch_data: 一个列表，装有batch_size个样本的数据字典，待pad
        :param i: batch索引号
        :return: pad完毕后，一个batch的数据字典
        """

        batch_text = []
        batch_contra_pos = []
        batch_polarity = []
        batch_dependency_graph = []
        batch_text_len = []
        max_len = max([t[self.sort_key].size(0) for t in batch_data])
        # if max_len != 7:
        #     return []
        # if max_len == 7:
        #     print([len(t[self.sort_key]) for t in batch_data])
        # for item in batch_data:
        #     if len(item['dependency_graph'])==8:
        #         print(item['text'])

        for item in batch_data:
            text, contra_pos , polarity, dependency_graph = \
                item['text'], item['contra_pos'], \
                item['polarity'], item['dependency_graph']
            text_len = text.size(0)
            batch_text_len.append(text_len)
            pad = torch.zeros([max_len-text_len,768])
            text = torch.cat([text,pad], dim=0)
            batch_text.append(text)
            batch_polarity.append(polarity)
            batch_contra_pos.append(contra_pos)
            batch_dependency_graph.append(numpy.pad(dependency_graph, \
                ((0,max_len-text_len),(0,max_len-text_len)), 'constant'))


        #pdb.set_trace()
        #print(type(embeddings[i]['elmo_representations'][0]))
        return {
                'text_bert': torch.cat(batch_text,dim=0).view([len(batch_data), -1, 768]),          #32, maxlen, 768
                'batch_text_len':torch.tensor(batch_text_len), #32,
                'contra_pos': batch_contra_pos,
                'polarity': torch.tensor(batch_polarity),      #32,
                'dependency_graph': torch.tensor(batch_dependency_graph)
                    }

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.batches)
        for idx in range(self.batch_len):
            yield self.batches[idx]
