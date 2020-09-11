
# code for 2021 AAAI paper
## Sarcasm Detection with Sentiment-inconsistency Graph Attention Network
## requirments:
+ PyTorch 1.5.1
+ nltk(sentiwordnet) 3.5
+ SpaCy 2.0.0
+ pytorch_pretrained_bert 0.6.2

## Quick Start
1. CUDA_VISIBLE_DEVICES=0 python3 data_prepeocess.py //generate BERT word Embedding & sentence graph
2. CUDA_VISIBLE_DEVICES=0 python3 train.py --batch_size=32 --save=1
