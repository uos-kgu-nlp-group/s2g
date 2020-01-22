# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 16:55:04 2020

@author: mayson
"""

#%%
import tensorflow as tf
from tensorflow.keras import layers
#from tensorflow.keras import models
from tensorflow.keras import preprocessing
print('tensorflow version: {}'.format(tf.__version__))
print('즉시 실행 모드: {}'.format(tf.executing_eagerly()))

#%%
#import json
import pandas as pd
import numpy as np
from pprint import pprint
from progressbar import progressbar

default_path = r'C:\Users\dpelt\Desktop\Mayson\UOS_graduate\s2g'

#%%
# parallel corpus
s2g_data = pd.read_csv(default_path + '/result/s2g_train_tokenized.csv',
                       encoding='utf-8',
                       sep=',')

#%%
# partitioning: source & target
source_corpus = s2g_data.Q
target_corpus = s2g_data.C.apply(lambda x: '<sos> ' + x + ' <eos>')
pprint(source_corpus[:10])
pprint(target_corpus[:10])

#%%
# source & target vocab 준비
source_vocab = set()
for i in progressbar(range(len(source_corpus))):
    source_vocab.update(source_corpus[i].split())
source_vocab = {x : i+1 for i, x in enumerate(list(source_vocab))}
source_vocab['<UNK>'] = 0
source_vocab_size = len(source_vocab)

target_vocab = set()
for i in progressbar(range(len(target_corpus))):
    target_vocab.update(target_corpus[i].split())
target_vocab = {x : i+1 for i, x in enumerate(list(target_vocab))}
target_vocab['<UNK>'] = 0
target_vocab_size = len(target_vocab)

print('source vocab size: {}'.format(source_vocab_size))
print('target vocab size: {}'.format(target_vocab_size))

#%%
# vocab 저장
with open(default_path + '/result/source_vocab.txt', 'w', encoding='utf-8') as f:
    for x, y in list(source_vocab.items()):
        f.write(x + '\n')

with open(default_path + '/result/target_vocab.txt', 'w', encoding='utf-8') as f:
    for x, y in list(target_vocab.items()):
        f.write(x + '\n')

#%%
# index로 이루어진 sequence 준비
source_sequences = []
for i in progressbar(range(len(source_corpus))):
    line = source_corpus[i]
    temp = []
    for word in line.split():
        temp.append(source_vocab.get(word))
    source_sequences.append(temp)
print(source_sequences[:10])

target_sequences = []
for i in progressbar(range(len(target_corpus))):
    line = target_corpus[i]
    temp = []
    for word in line.split():
        temp.append(target_vocab.get(word))
    target_sequences.append(temp)
print(target_sequences[:10])

# target sequence와 비교할 true sequence 준비 -> target_sequence에서 <sos>를 제거한다
true_sequences = []
for i in progressbar(range(len(target_corpus))):
    line = target_corpus[i]
    temp = []
    flag = 0
    for word in line.split():
        if flag:
            temp.append(target_vocab.get(word))
        flag += 1
    true_sequences.append(temp)
print(true_sequences[:10])

#%%
# padding
max_source_len = max([len(x) for x in source_sequences])
max_target_len = max([len(x) for x in target_sequences])
print(max_source_len)
print(max_target_len)

source_input = preprocessing.sequence.pad_sequences(source_sequences,
                                                    maxlen=max_source_len,
                                                    padding='post')
target_input = preprocessing.sequence.pad_sequences(target_sequences,
                                                    maxlen=max_target_len,
                                                    padding='post')

# true categorical data for learning
true_output = preprocessing.sequence.pad_sequences(true_sequences,
                                                    maxlen=max_target_len,
                                                    padding='post')

from tensorflow.keras.utils import to_categorical
true_output = to_categorical(true_output)

#%%
# seq2seq
embedding_size = 256

encoder_inputs = layers.Input(shape=(max_source_len, ))
encoder_embedding_layer = layers.Embedding(input_dim=source_vocab_size, 
                                           output_dim=embedding_size)
encoder_embedded = encoder_embedding_layer(encoder_inputs)
encoder_lstm = layers.LSTM(units=128,
                           return_state=True,
                           name='encoder')
encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedded)
encoder_states = [state_h, state_c]

decoder_inputs = layers.Input(shape=(max_target_len, ))
decoder_embedding_layer = layers.Embedding(input_dim=target_vocab_size, 
                                           output_dim=embedding_size)
decoder_embedded = decoder_embedding_layer(decoder_inputs)
decoder_lstm = layers.LSTM(units=128,
                           return_sequences=True,
                           return_state=True,
                           name='decoder')
decoder_outputs, _, _ = decoder_lstm(decoder_embedded, initial_state=encoder_states)
decoder_softmax = layers.Dense(target_vocab_size,
                               activation='softmax')
outputs = decoder_softmax(decoder_outputs)

model = tf.keras.Model([encoder_inputs, decoder_inputs], outputs)
model.summary()

#%%
# training
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy')

model.fit(x=[source_input, target_input], y=true_output,
          batch_size=64,
          epochs=10,
          validation_split=0.2)

#%%
# prediction model setting
# left part
encoder_model = tf.keras.Model(inputs=encoder_inputs, outputs=encoder_states) 

# right part
decoder_state_input_h = layers.Input(shape=(128, ))
decoder_state_input_c = layers.Input(shape=(128, ))
decoder_state_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_embedded = decoder_embedding_layer(decoder_inputs)
decoder_outputs, state_h, state_c = decoder_lstm(decoder_embedded, 
                                                 initial_state=decoder_state_inputs)
decoder_states_output = [state_h, state_c]
softmax_outputs = decoder_softmax(decoder_outputs)

decoder_model = tf.keras.Model(inputs=[decoder_inputs] + decoder_state_inputs,
                               outputs=[softmax_outputs] + decoder_states_output)

#%%
# index 2 word dict
source_idx2word = {i : word for word, i in source_vocab.items()}
target_idx2word = {i : word for word, i in target_vocab.items()}

#%%    
def decode_sequence(sequence):
    input_seq = np.array(sequence).reshape((1, 11))
    
    # encoder의 마지막 hidden state 값을 저장
    last_hidden_states = encoder_model.predict(input_seq)
    # <sos>에 해당하는 input 생성
    target_seq = np.zeros((1, max_target_len))
    target_seq[0, 0] = target_vocab.get('<sos>')
    
    stop_condition = False
    decoded_sequence = []
    idx = 0
    while not stop_condition:
        output_subwords, _, _ = decoder_model.predict([target_seq] + last_hidden_states)
        max_vocab_index = np.argmax(output_subwords[0][idx])
        argmax_subword = target_idx2word.get(max_vocab_index)
        decoded_sequence.append(argmax_subword)
        
        if argmax_subword == '<eos>' or len(decoded_sequence) > max_target_len:
            stop_condition = True
        
        idx += 1
        target_seq[0, idx] = max_vocab_index
    
    return decoded_sequence

#%%
# parallel corpus for test
s2g_test = pd.read_csv(default_path + '/result/s2g_test_tokenized.csv',
                        encoding='utf-8',
                        sep=',')
source_test = s2g_test.Q
pprint(source_test[:10])

#%%
test_sample = '요즘 짠한 남자 종로구 그래프'
encoded = tokenizer.encode(test_sample)
test = [source_vocab.get(x, 0) for x in encoded.tokens]
test1 = preprocessing.sequence.pad_sequences([test], 
                                     maxlen=max_source_len,
                                     padding='post')
' '.join(decode_sequence(test1)).replace('▁', '')

#%%
# index로 이루어진 sequence for test
source_test_sequences = []
for i in progressbar(range(len(source_test))):
    line = source_test[i]
    temp = []
    for word in line.split():
        temp.append(source_vocab.get(word, 0)) # 없으면 <UNK> 처리
    source_test_sequences.append(temp)
print(source_test_sequences[:10])

#%%
# padding for test
source_test_input = preprocessing.sequence.pad_sequences(source_test_sequences,
                                                         maxlen=max_source_len,
                                                         padding='post')

#%%
# decoding for test data
decoded_result = []
for i in progressbar(range(len(source_test_input))):
    decoded_result.append(decode_sequence(source_test_input[i]))
pprint(decoded_result[:10])
decoded_refined = [' '.join(x).replace('▁', '').replace(' <eos>', '') for x in decoded_result]

#%%
# 저장
source_refined = [x.replace('▁', '') for x in source_test]
final_save = pd.concat([pd.DataFrame({'Q': source_refined}), pd.DataFrame({'C': decoded_refined})], axis=1)
final_save.to_csv(default_path + '/result/s2g_final_result.csv',
                  sep=',',
                  index=False,
                  header=True,
                  encoding='cp949')

#%%







































