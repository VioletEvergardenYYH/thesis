import numpy as np
import spacy
import nltk
import pickle
import pdb
from nltk.tokenize import TweetTokenizer
import re

tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True)
nlp = spacy.load('en_core_web_sm')
trainA = './datasets/train/SemEval2018-T3-train-taskA.txt'
trialA = './datasets/trial/example-dataset-taskA.txt'
testA_ = './datasets/test_TaskA/SemEval2018-T3_input_test_taskA.txt'
testA = './datasets/goldtest_TaskA/SemEval2018-T3_gold_test_taskA_emoji.txt'
examA = '.datasets/trial/example-dataset-taskA.txt'


def test(t):
    doc = nlp(t)
    seq_len = len(t.split())
    for token in doc:
        print('//' + str(token))
        for child in token.children:
            print(child.text)
def make_corpus(fp):
    '''
    Loads the dataset .txt file with label-tweet on each line and parses the dataset.
    :param fp: filepath of dataset
    :return:
        corpus: list of tweet strings of each tweet.
        y: list of labels
    '''
    corpus = []
    with open(fp, 'rt', encoding='utf-8') as data_in:
        for line in data_in:
            if not line.lower().startswith("tweet index"): # discard first line if it contains metadata
                line = line.rstrip() # remove trailing whitespace
                if fp.count('test'):
                    tweet_raw = line.split('\t')[1]
                    l = line.split('\t')[0]
                else:
                    tweet_raw = line.split("\t")[2]
                    l = line.split('\t')[1]
                tweet = tokenizer.tokenize(tweet_raw)
                # if len(tweet)<3:
                #     print(l+tweet_raw)
                corpus.append(tweet)
    return corpus

def dependency_adj_matrix(text):
    # https://spacy.io/docs/usage/processing-text
    document = nlp(text)
    seq_len = len(text.split())

    matrix = np.zeros((seq_len, seq_len)).astype('float32')
    for token in document:
        #print(token)
        if token.i < seq_len:
            matrix[token.i][token.i] = 1  # 自连接
            # https://spacy.io/docs/api/token
            for child in token.children:
                if child.i < seq_len:
                    matrix[token.i][child.i] = 1
                    matrix[child.i][token.i] = 1
    #print(matrix)
    #pdb.set_trace()
    return matrix


# def process(filename):
#     c = make_corpus(filename)
#     print(len(c))
#     idx2graph = {}
#     fout = open(filename + '.graph', 'wb')
#     for i in range(len(c)):
#         adj_matrix = dependency_adj_matrix(' '.join(c[i]))
#         idx2graph[i+1] = adj_matrix
#     #print(idx2graph)
#     pickle.dump(idx2graph, fout)
#     fout.close()

def bert_process(filename):
    f = open(filename, 'rb')
    data = pickle.load(f)
    f.close()
    idx2graph = {}
    fout = open(filename + '.graph', 'wb')
    for id, v in data.items():
        v[0][0] = 'CLS'
        v[0][-1] = 'SEP'
        text = ' '.join(v[0])
        matrix = dependency_adj_matrix(text)
        idx2graph[id] = matrix
    pickle.dump(idx2graph, fout)
    fout.close()



if __name__ == '__main__':
    # make_corpus(trainA)
    # process(testA)
    # process(trainA)
    #adj_matrix = dependency_adj_matrix(' '.join(["you're", 'great', 'at', 'keeping', 'a', 'conversation', '. .']))
    bert_process('./datasets/goldtest_TaskA/test_hash.pkl')
    bert_process('./datasets/train/train_hash.pkl')
    #bug: 邻接矩阵函数将被tweet分词器视为一个词的. .视作两个词
    #核心问题：生成word2idx的分词器没见过. .这个词，使用text_indice不会出错是因为，由普通分词器生成的text_indice将. .视作两个词
