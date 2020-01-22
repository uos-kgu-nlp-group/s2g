# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 13:53:18 2020

@author: mayson
"""
#%%
import pandas as pd
import time
import json
from pprint import pprint
from progressbar import progressbar
#from tokenizers import BPETokenizer
#from tokenizers import BertWordPieceTokenizer
from tokenizers import SentencePieceBPETokenizer

default_path = r'C:\Users\dpelt\Desktop\Mayson\UOS_graduate\s2g'

#%%
raw_s2g_data = pd.read_csv(default_path + '/speech_to_ggplot_data_2.csv',
                        encoding='utf-8',
                        sep=',')
raw_s2g_data = raw_s2g_data.sample(frac=1).reset_index(drop=True)

s2g_train = raw_s2g_data['Q'][:-1000]
s2g_train = s2g_train.append(raw_s2g_data['C'][:-1000])
s2g_test = raw_s2g_data['Q'][-1000:]
s2g_test = s2g_test.append(raw_s2g_data['C'][-1000:])

with open(default_path + '/s2g_train.txt', 'w', encoding='utf-8') as f:
    for i in range(len(s2g_train)):
        f.write(s2g_train.iloc[i]+ '\n')
        
with open(default_path + '/s2g_test.txt', 'w', encoding='utf-8') as f:
    for i in range(len(s2g_test)):
        f.write(s2g_test.iloc[i]+ '\n')
        
#%%
# Initialize a tokenizer
tokenizer = SentencePieceBPETokenizer()

# Then train it!
# only work fast for utf-8!***
start = time.time()
tokenizer.train([ default_path + '/s2g_train.txt' ])
print('Training time : {}'.format(time.time() - start))

#%%
# train data에 대한 tokenizer 학습 & tokenized 결과 저장
train_s2g = []
with open(default_path + '/s2g_train.txt', 'r', encoding='utf-8') as f:
    while True:
        line = f.readline()
        if not line: break
        train_s2g.append(line.replace('\n', ''))
pprint(train_s2g[-10:])

train_s2g_tokenized = []
for i in progressbar(range(len(train_s2g))):
    encoded = tokenizer.encode(train_s2g[i])
    train_s2g_tokenized.append(' '.join(encoded.tokens))
pprint(train_s2g_tokenized[-10:])

save_file_train = pd.concat([pd.DataFrame({'Q': train_s2g_tokenized[:int(len(train_s2g_tokenized) / 2)]}), 
                             pd.DataFrame({'C': train_s2g_tokenized[int(len(train_s2g_tokenized) / 2):]})], axis=1)
save_file_train.to_csv(default_path + '/result/s2g_train_tokenized.csv',
                         sep=',',
                         index=False,
                         header=True,
                         encoding='utf-8')

#%%
# test data에 대한 tokenizer 적용 & 저장
test_s2g = []
with open(default_path + '/s2g_test.txt', 'r', encoding='utf-8') as f:
    while True:
        line = f.readline()
        if not line: break
        test_s2g.append(line.replace('\n', ''))
pprint(test_s2g[-10:])

test_s2g_tokenized = []
for i in progressbar(range(len(test_s2g))):
    encoded = tokenizer.encode(test_s2g[i])
    test_s2g_tokenized.append(' '.join(encoded.tokens))
pprint(test_s2g_tokenized[-10:])

save_file_test = pd.concat([pd.DataFrame({'Q': test_s2g_tokenized[:int(len(test_s2g_tokenized) / 2)]}), 
                            pd.DataFrame({'C': test_s2g_tokenized[int(len(test_s2g_tokenized) / 2):]})], axis=1)
save_file_test.to_csv(default_path + '/result/s2g_test_tokenized.csv',
                         sep=',',
                         index=False,
                         header=True,
                         encoding='utf-8')

#%%
# =============================================================================
# 여기까지 사용
# =============================================================================
#%%
# And finally save it somewhere -> save merge steps and vocabulary
tokenizer.save(default_path + '/result', "s2g_config")

#%%
with open(default_path + '/result/s2g_config-vocab.json', 'r', encoding='utf-8') as f:
    vocab = json.load(f)
print(sorted(list(vocab.items()), key=lambda x: -x[1])[:10]) # print vocab of codes

#%%





















