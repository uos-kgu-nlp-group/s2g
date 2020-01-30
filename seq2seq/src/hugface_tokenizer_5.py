# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 16:16:52 2020

@author: mayson
"""


#%%
import pandas as pd
import time
import json
import csv
from pprint import pprint
#import random
from progressbar import progressbar
# tokenizer
#from tokenizers import BPETokenizer
#from tokenizers import BertWordPieceTokenizer
from tokenizers import SentencePieceBPETokenizer
#from tokenizers import Tokenizer

default_path = r'C:\Users\dpelt\Desktop\Mayson\UOS_graduate\s2g'
version_num = 5

#%%
raw_s2g_data = pd.read_csv(default_path + '/data/pop_dat_fin.csv',
                           encoding='utf-8',
                           sep=',')

raw_s2g_data = raw_s2g_data.sample(frac=1).reset_index(drop=True) # shuffling data

s2g_train = raw_s2g_data['Q'][:-1000]
s2g_train = s2g_train.append(raw_s2g_data['C'][:-1000])

with open(default_path + '/data/s2g_train_{}.txt'.format(version_num), 'w', encoding='utf-8') as f:
    for i in range(len(s2g_train)):
        f.write(s2g_train.iloc[i] + '\n')

pd.DataFrame({'Q': raw_s2g_data['Q'][-1000:], 'C': raw_s2g_data['C'][-1000:]}).to_csv(default_path + '/data/s2g_test_{}.csv'.format(version_num),
                       sep=',',
                       index=False,
                       header=True,
                       encoding='utf-8')

#%%
# =============================================================================
# start from here if data is prepared
# =============================================================================
        
# Initialize a tokenizer
tokenizer = SentencePieceBPETokenizer()

# train tokenizer
start = time.time()

#tokenizer.train([default_path + '/data/s2g_train_{}.txt'.format(version_num)], # only work fast for utf-8!***
#                vocab_size=32000,
#                min_frequency=3,
#                special_tokens=["<UNK>"])

tokenizer.train([default_path + '/data/s2g_train_{}.txt'.format(version_num)]) # only work fast for utf-8!***
print('Training time : {}'.format(time.time() - start))

#%%
# train data에 대한 tokenize & tokenzied 결과 저장
s2g_train = []
with open(default_path + '/data/s2g_train_{}.txt'.format(version_num), 'r', encoding='utf-8') as f:
    while True:
        line = f.readline()
        if not line: break
        s2g_train.append(line.replace('\n', ''))
pprint(s2g_train[-10:])

s2g_train_tokenized = []
for i in progressbar(range(len(s2g_train))):
    encoded = tokenizer.encode(s2g_train[i])
    s2g_train_tokenized.append(' '.join(encoded.tokens).replace('▁', ''))
pprint(s2g_train_tokenized[-10:])

save_file_train = pd.concat([pd.DataFrame({'Q': s2g_train_tokenized[:int(len(s2g_train_tokenized) / 2)]}), 
                             pd.DataFrame({'C': s2g_train_tokenized[int(len(s2g_train_tokenized) / 2):]})], axis=1)
save_file_train.to_csv(default_path + '/result/s2g_train_tokenized_{}.csv'.format(version_num),
                       sep=',',
                       index=False,
                       header=True,
                       encoding='utf-8')

#%%
# =============================================================================
# 여기까지 사용
# =============================================================================
# And finally save it somewhere -> save merge steps and vocabulary
tokenizer.save(default_path + '/result', "s2g_config_{}".format(version_num))

#%%
with open(default_path + '/result/s2g_config_{}-vocab.json'.format(version_num), 'r', encoding='utf-8') as f:
    vocab = json.load(f)
print(sorted(list(vocab.items()), key=lambda x: -x[1])[:10]) # print vocab of codes

#%%













































