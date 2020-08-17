import os
import pickle
import numpy as np
import nltk
from nltk.tokenize import TweetTokenizer
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from nltk.corpus import sentiwordnet as swn

def load_word_vec(path, word2idx=None):
    fin = open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')
    word_vec = {}
    for line in fin:
        tokens = line.rstrip().split()
        if word2idx is None or tokens[0] in word2idx.keys():
            try:
                word_vec[tokens[0]] = np.asarray(tokens[1:], dtype='float32')
            except:
                continue
    return word_vec


def build_embedding_matrix(word2idx, embed_dim, type):
    embedding_matrix_file_name = '{0}_{1}_embedding_matrix.pkl'.format(str(embed_dim), type)
    if os.path.exists(embedding_matrix_file_name):
        print('loading embedding_matrix:', embedding_matrix_file_name)
        embedding_matrix = pickle.load(open(embedding_matrix_file_name, 'rb'))
    else:
        print('loading word vectors ...')
        embedding_matrix = np.zeros((len(word2idx), embed_dim))  # idx 0 and 1 are all-zeros
        embedding_matrix[1, :] = np.random.uniform(-1/np.sqrt(embed_dim), 1/np.sqrt(embed_dim), (1, embed_dim))
        fname = 'E:/glove/glove.840B.300d.txt'
        word_vec = load_word_vec(fname, word2idx=word2idx)
        print('building embedding_matrix:', embedding_matrix_file_name)
        for word, i in word2idx.items():
            vec = word_vec.get(word)
            if vec is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = vec
        pickle.dump(embedding_matrix, open(embedding_matrix_file_name, 'wb'))
    return embedding_matrix


class Tokenizer(object):
    def __init__(self, word2idx=None):
        if word2idx is None:
            self.word2idx = {}
            self.idx2word = {}
            self.idx = 0
            self.word2idx['<pad>'] = self.idx
            self.idx2word[self.idx] = '<pad>'
            self.idx += 1
            self.word2idx['<unk>'] = self.idx
            self.idx2word[self.idx] = '<unk>'
            self.idx += 1
        else:
            self.word2idx = word2idx
            self.idx2word = {v:k for k,v in word2idx.items()}

    def fit_on_text(self, text):
        text = text.lower()
        words = text.split()
        for word in words:
            if word not in self.word2idx:
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx += 1

    def text_to_sequence(self, text):
        text = text.lower()
        words = text.split()
        unknownidx = 1
        sequence = [self.word2idx[w] if w in self.word2idx else unknownidx for w in words]
        if len(sequence) == 0:
            sequence = [0]
        return sequence


class IDDataset(object):
    """
    Irony Detection Dataset
    用于封装整体数据，增加切片索引功能
    """
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)



class IDDatesetReader:
    @staticmethod
    def __read_text__(fnames):   #文本预处理
        text = ''
        tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True)
        corpus = ''
        for fname in fnames:
            fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
            lines = fin.readlines()
            fin.close()

            for line in lines:
                if not line.lower().startswith("tweet index"):  # ignore column names
                    line = line.rstrip()
                    if fname.count('test'):
                        tweet = line.split("\t")[1]
                    else:
                        tweet = line.split("\t")[2]

                    corpus += tweet + ' '

        text = tokenizer.tokenize(corpus)
        return ' '.join(text)

    @staticmethod
    def __read_bert_data__(fname):
        fin = open(fname, 'rb')
        data = pickle.load(fin)
        fin.close()
        fin = open(fname+'.graph', 'rb')
        graph = pickle.load(fin)
        fin.close()
        all_data = []
        for id, v in data.items():
            text = v[0]
            polarity = int(v[1])
            text_tensor = v[2]
            dependency_graph = graph[id]
            contra_pos = []

            pos_all = []
            neg_all = []
            for i in range(len(text)):  # 每个词
                p_score = 0
                n_score = 0
                t = list(swn.senti_synsets(text[i]))
                if len(t):
                    for w in t:
                        p_score += w.pos_score()
                        n_score += w.neg_score()
                    p_score /= len(t)
                    n_score /= len(t)
                    pos_all.append(p_score)
                    neg_all.append(n_score)
                else:
                    pos_all.append(-1)
                    neg_all.append(-1)
            m_pos = max(pos_all)
            m_neg = max(neg_all)

            contra_pos.append(pos_all.index(m_pos))
            contra_pos.append(neg_all.index(m_neg))

            dic = {
                'text': text_tensor,
                'contra_pos': contra_pos,
                'polarity': polarity,
                'dependency_graph': dependency_graph
            }
            all_data.append(dic)


        return all_data






    @staticmethod
    def __read_elmo_data__(fname):
        t_tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True)
        test_label = {}
        fin = open('./datasets/goldtest_TaskA/SemEval2018-T3_gold_test_taskA_emoji.txt', 'r', encoding='utf-8',
                   newline='\n', errors='ignore')
        lines = fin.readlines()
        for line in lines:
            if not line.lower().startswith("tweet index"):
                line = line.rstrip()
                test_label[int(line.split('\t')[0])] = (int(line.split('\t')[1]))
        fin.close()
        fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        lines = fin.readlines()
        fin.close()

        fin = open(fname + '.graph', 'rb')
        idx2gragh = pickle.load(fin)
        print(len(idx2gragh))
        fin.close()

        all_data = []
        for line in lines:  # 每句话
            if not line.lower().startswith("tweet index"):
                contra_pos = []
                line = line.rstrip()
                if fname.count('test'):
                    text = line.split('\t')[1]
                    polarity = test_label[int(line.split('\t')[0])]
                else:
                    text = line.split('\t')[2]
                    polarity = int(line.split("\t")[1])
                text = t_tokenizer.tokenize(text)
                text = ' '.join(text)
                text = text.split()
                # if len(text) < 3:
                #     continue
                dependency_graph = idx2gragh[int(line.split('\t')[0])]
                #text_indices = tokenizer.text_to_sequence(' '.join(text))  # 返回列表

                pos_all = []
                neg_all = []
                for i in range(len(text)):  # 每个词
                    p_score = 0
                    n_score = 0
                    t = list(swn.senti_synsets(text[i]))
                    if len(t):
                        for w in t:
                            p_score += w.pos_score()
                            n_score += w.neg_score()
                        p_score /= len(t)
                        n_score /= len(t)
                        pos_all.append(p_score)
                        neg_all.append(n_score)
                    else:
                        pos_all.append(-1)
                        neg_all.append(-1)
                m_pos = max(pos_all)
                m_neg = max(neg_all)

                contra_pos.append(pos_all.index(m_pos))
                contra_pos.append(neg_all.index(m_neg))
                data = {
                    'text': text,   #列表，装着token
                    'contra_pos': contra_pos,  # 冲突词位置
                    'polarity': polarity,
                    'dependency_graph': dependency_graph,
                }

                all_data.append(data)
        return all_data
    @staticmethod
    def __read_data__(fname, tokenizer):
        t_tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True)
        test_label={}
        fin = open('./datasets/goldtest_TaskA/SemEval2018-T3_gold_test_taskA_emoji.txt', 'r', encoding='utf-8', newline='\n', errors='ignore')
        lines = fin.readlines()
        for line in lines:
            if not line.lower().startswith("tweet index"):
                line = line.rstrip()
                test_label[int(line.split('\t')[0])] = (int(line.split('\t')[1]))
        fin.close()
        fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        lines = fin.readlines()
        fin.close()

        fin = open(fname+'.graph', 'rb')
        idx2gragh = pickle.load(fin)
        print(len(idx2gragh))
        fin.close()
        all_data = []
        for line in lines:   #每句话
            if not line.lower().startswith("tweet index"):
                contra_pos = []
                line = line.rstrip()
                if fname.count('test'):
                    text = line.split('\t')[1]
                    polarity = test_label[int(line.split('\t')[0])]
                else:
                    text = line.split('\t')[2]
                    polarity = int(line.split("\t")[1])
                text = t_tokenizer.tokenize(text)
                if len(text)<3:
                    continue
                dependency_graph = idx2gragh[int(line.split('\t')[0])]
                text_indices = tokenizer.text_to_sequence(' '.join(text))  #返回列表
                pos_all = []
                neg_all = []
                for i in range(len(text)):    #每个词
                    p_score = 0
                    n_score = 0
                    t = list(swn.senti_synsets(text[i]))
                    if len(t):
                        for w in t:
                            p_score += w.pos_score()
                            n_score += w.neg_score()
                        p_score /= len(t)
                        n_score /= len(t)
                        pos_all.append(p_score)
                        neg_all.append(n_score)
                    else:
                        pos_all.append(-1)
                        neg_all.append(-1)
                m_pos = max(pos_all)
                m_neg = max(neg_all)

                contra_pos.append(pos_all.index(m_pos))
                contra_pos.append(neg_all.index(m_neg))
                data = {
                    'text_indices': text_indices,
                    'contra_pos': contra_pos,  # 冲突词位置
                    'polarity': polarity,
                    'dependency_graph': dependency_graph,
                }

                all_data.append(data)
        return all_data

    def __init__(self, dataset='twitter', embed_dim=300):
        print("preparing {0} dataset ...".format(dataset))
        fname = {
            'twitter': {
                'train': './datasets/train/train.pkl',
                'example': 'datasets/trial/trial.pkl',
                'test': './datasets/goldtest_TaskA/test.pkl'
            },
            'emoji': {
                'train': './datasets/train/SemEval2018-T3-train-taskA_emoji.txt',
                'test': './datasets/test_TaskA/SemEval2018-T3_input_test_taskA_emoji.txt'
            },
            'hashtag': {
                'train': './datasets/train/SemEval2018-T3-train-taskA_emoji_ironyHashtags.txt',
                'test': './datasets/test_TaskA/SemEval2018-T3_input_test_taskA_emoji.txt'
            },


        }
        # text = IDDatesetReader.__read_text__([fname[dataset]['train'], fname[dataset]['test']])
        # if os.path.exists(dataset+'_word2idx.pkl'):
        #     print("loading {0} tokenizer...".format(dataset))
        #     with open(dataset+'_word2idx.pkl', 'rb') as f:
        #          word2idx = pickle.load(f)
        #          tokenizer = Tokenizer(word2idx=word2idx)
        # else:
        #     tokenizer = Tokenizer()
        #     tokenizer.fit_on_text(text)
        #     with open(dataset+'_word2idx.pkl', 'wb') as f:
        #          pickle.dump(tokenizer.word2idx, f)
        # self.embedding_matrix = build_embedding_matrix(tokenizer.word2idx, embed_dim, dataset)
        self.train_data = IDDataset(IDDatesetReader.__read_bert_data__(fname[dataset]['train']))
        #列表，装有每个样本对应的数据字典
        self.test_data = IDDataset(IDDatesetReader.__read_bert_data__(fname[dataset]['test']))

if __name__ == '__main__':

    id_dataset = IDDatesetReader()
    data = id_dataset.train_data
    print(len(data))
    print(data[0])
