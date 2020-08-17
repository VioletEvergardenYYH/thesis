import torch
import re
import pickle
import copy
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
def wash_text(text):
    text = text.split()
    final = []
    for t in text:
        if re.search('@|http',t):
            pass
        else:
            final.append(t.lower())
    return ' '.join(final)

def process(path):
    """
    :param path:
    得到一个数据字典：        data_all[id] = [tokenized_text, label, summed_last_4_layers]，并将其dump成pkl文件
    tokenized_text：       [str, str, str]，每个str是一个token
    summed_last_4_layers： [number_of_tokens, 768]
    """
    data_all = {}
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    f = open(path, 'r')
    next(f)
    for line in f:
        line = line.strip().split('\t')
        id, label, text = line
        print(id)
        text = wash_text(text)
        marked_text = "[CLS] " + text + " [SEP]"
        tokenized_text = tokenizer.tokenize(marked_text) #[str, str, str]

        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text) #[int, int , int]
        # if id == '13' or id == '28':
        #     print(indexed_tokens)
        #     print(tokenized_text)

        segments_ids = [1] * len(indexed_tokens)

        tokens_tensor = torch.tensor([indexed_tokens])

        segments_tensors = torch.tensor([segments_ids])
        # Load pre-trained model (weights)
        #data_all[id] = [tokenized_text, label, tokens_tensor, segments_tensors]

        model = BertModel.from_pretrained('bert-base-uncased')
        # Put the model in "evaluation" mode, meaning feed-forward operation.
        model.eval()
        # Predict hidden states features for each layer
        with torch.no_grad():
            encoded_layers, _ = model(tokens_tensor, segments_tensors)

        token_embeddings = []
        # For each token in the sentence...
        for token_i in range(len(tokenized_text)):
          # Holds 12 layers of hidden states for each token
          hidden_layers = []
          # For each of the 12 layers...
          for layer_i in range(len(encoded_layers)):
            # Lookup the vector for `token_i` in `layer_i`
            vec = encoded_layers[layer_i][0][token_i]
            hidden_layers.append(vec)
          token_embeddings.append(hidden_layers)

        summed_last_4_layers = torch.cat([torch.sum(torch.stack(layer)[-4:], 0) for layer in token_embeddings],dim=0).view([-1,768]) # [number_of_tokens, 768]
        data_all[id] = [tokenized_text, label, summed_last_4_layers]
    with open('./datasets/goldtest_TaskA/test_hash.pkl', 'wb') as f:
        pickle.dump(data_all, f)



if __name__ == '__main__':
    process('./datasets/goldtest_TaskA/SemEval2018-T3_gold_test_taskA_emoji.txt')



