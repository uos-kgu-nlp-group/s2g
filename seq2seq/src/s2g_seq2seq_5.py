# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 16:25:12 2020

@author: mayson
"""


'''
4 models
'''
#%%
import tensorflow as tf
from tensorflow.keras import layers
#from tensorflow.keras import models
from tensorflow.keras import preprocessing
from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_json
print('tensorflow version: {}'.format(tf.__version__))
print('즉시 실행 모드: {}'.format(tf.executing_eagerly()))

#%%
import pandas as pd
import numpy as np
from pprint import pprint
from progressbar import progressbar

default_path = r'C:\Users\dpelt\Desktop\Mayson\UOS_graduate\s2g'
version_num = 5

#%%
# parallel corpus
s2g_data = pd.read_csv(default_path + '/result/s2g_train_tokenized_{}.csv'.format(version_num),
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
source_vocab = {x : i+2 for i, x in enumerate(list(source_vocab))}
source_vocab['<PAD>'] = 0
source_vocab['<UNK>'] = 1
source_vocab_size = len(source_vocab)

target_vocab = set()
for i in progressbar(range(len(target_corpus))):
    target_vocab.update(target_corpus[i].split())
target_vocab = {x : i+2 for i, x in enumerate(list(target_vocab))}
target_vocab['<PAD>'] = 0
target_vocab['<UNK>'] = 1
target_vocab_size = len(target_vocab)

print('source vocab size: {}'.format(source_vocab_size))
print('target vocab size: {}'.format(target_vocab_size))

# index 2 word dict
source_idx2word = {i : word for word, i in source_vocab.items()}
target_idx2word = {i : word for word, i in target_vocab.items()}

#%%
# vocab 저장
with open(default_path + '/result/source_vocab_{}.txt'.format(version_num), 'w', encoding='utf-8') as f:
    for x, y in list(source_vocab.items()):
        f.write(x + '\n')

with open(default_path + '/result/target_vocab_{}.txt'.format(version_num), 'w', encoding='utf-8') as f:
    for x, y in list(target_vocab.items()):
        f.write(x + '\n')
        
#%%
# index로 이루어진 sequence 준비
source_sequences = []
for i in progressbar(range(len(source_corpus))):
    line = source_corpus[i]
    temp = []
    for word in line.split():
        temp.append(source_vocab.get(word, 1))
    source_sequences.append(temp)
print(source_sequences[:10])

target_sequences = []
for i in progressbar(range(len(target_corpus))):
    line = target_corpus[i]
    temp = []
    for word in line.split():
        temp.append(target_vocab.get(word, 1))
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
            temp.append(target_vocab.get(word, 1))
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
'''
model #1: shallow LSTMs
'''
embedding_size = 256
deep = False
bidirectional = True # choose bidirectional

encoder_inputs = layers.Input(shape=(max_source_len, ))
encoder_embedding_layer = layers.Embedding(input_dim=source_vocab_size, 
                                           output_dim=embedding_size)
encoder_embedded = encoder_embedding_layer(encoder_inputs)
if bidirectional:
    encoder_lstm = layers.Bidirectional(layers.LSTM(units=128,
                                                    return_state=True,
                                                    name='encoder'))
    encoder_outputs, forward_h, forward_c, backward_h, backward_c = encoder_lstm(encoder_embedded)
    state_h = layers.Concatenate()([forward_h, backward_h])
    state_c = layers.Concatenate()([forward_c, backward_c])
    encoder_states = [state_h, state_c]
else:
    encoder_lstm = layers.LSTM(units=128,
                               return_state=True,
                               name='encoder')
    encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedded)
    encoder_states = [state_h, state_c]    
    
decoder_inputs = layers.Input(shape=(max_target_len, ))
decoder_embedding_layer = layers.Embedding(input_dim=target_vocab_size, 
                                           output_dim=embedding_size)
decoder_embedded = decoder_embedding_layer(decoder_inputs)
        
decoder_lstm = layers.LSTM(units=(int(bidirectional) + 1) * 128,
                           return_sequences=True,
                           return_state=True,
                           name='decoder')
decoder_outputs, _, _ = decoder_lstm(decoder_embedded, initial_state=encoder_states)
decoder_softmax = layers.Dense(target_vocab_size,
                               activation='softmax')
outputs = decoder_softmax(decoder_outputs)
    
model = tf.keras.Model([encoder_inputs, decoder_inputs], outputs)
model.summary()

# training
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy')

model.fit(x=[source_input, target_input], y=true_output,
          batch_size=256,
          epochs=5,
          validation_split=0.2)

# prediction model setting
# left part
encoder_model = tf.keras.Model(inputs=encoder_inputs, outputs=encoder_states) 

# right part
decoder_state_input_h = layers.Input(shape=((int(bidirectional) + 1) * 128, ))
decoder_state_input_c = layers.Input(shape=((int(bidirectional) + 1) * 128, ))
decoder_state_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_embedded = decoder_embedding_layer(decoder_inputs)
decoder_outputs, state_h, state_c = decoder_lstm(decoder_embedded, 
                                                 initial_state=decoder_state_inputs)
#decoder_states_output = [state_h, state_c]
softmax_outputs = decoder_softmax(decoder_outputs)

#decoder_model = tf.keras.Model(inputs=[decoder_inputs] + decoder_state_inputs,
#                               outputs=[softmax_outputs] + decoder_states_output)

decoder_model = tf.keras.Model(inputs=[decoder_inputs] + decoder_state_inputs,
                               outputs=[softmax_outputs])
decoder_model.summary()

#%%
'''
model #2: deep LSTMs
'''
embedding_size = 256
deep = True
bidirectional = True # choose bidirectional

encoder_inputs = layers.Input(shape=(max_source_len, ))
encoder_embedding_layer = layers.Embedding(input_dim=source_vocab_size, 
                                           output_dim=embedding_size)
encoder_embedded = encoder_embedding_layer(encoder_inputs)
if bidirectional:
    encoder_lstm1 = layers.Bidirectional(layers.LSTM(units=128,
                                                     return_sequences=True,
                                                     return_state=True,
                                                     name='encoder_lstm1'))
    encoder_outputs1, forward_h1, forward_c1, backward_h1, backward_c1 = encoder_lstm1(encoder_embedded)
    encoder_lstm2 = layers.Bidirectional(layers.LSTM(units=128,
                                                     return_sequences=True,
                                                     return_state=True,
                                                     name='encoder_lstm2'))
    encoder_outputs2, forward_h2, forward_c2, backward_h2, backward_c2 = encoder_lstm2(encoder_outputs1)
    encoder_lstm3 = layers.Bidirectional(layers.LSTM(units=128,
                                                     return_sequences=True,
                                                     return_state=True,
                                                     name='encoder_lstm3'))
    encoder_outputs3, forward_h3, forward_c3, backward_h3, backward_c3 = encoder_lstm3(encoder_outputs2)
    state_h1 = layers.Concatenate()([forward_h1, backward_h1])
    state_c1 = layers.Concatenate()([forward_c1, backward_c1])
    state_h2 = layers.Concatenate()([forward_h2, backward_h2])
    state_c2 = layers.Concatenate()([forward_c2, backward_c2])
    state_h3 = layers.Concatenate()([forward_h3, backward_h3])
    state_c3 = layers.Concatenate()([forward_c3, backward_c3])
    
else:
    encoder_lstm1 = layers.LSTM(units=128,
                                return_sequences=True,
                                return_state=True,
                                name='encoder_lstm1')
    encoder_outputs1, state_h1, state_c1 = encoder_lstm1(encoder_embedded)
    encoder_lstm2 = layers.LSTM(units=128,
                                return_sequences=True,
                                return_state=True,
                                name='encoder_lstm2')
    encoder_outputs2, state_h2, state_c2 = encoder_lstm2(encoder_outputs1)
    encoder_lstm3 = layers.LSTM(units=128,
                                return_sequences=True,
                                return_state=True,
                                name='encoder_lstm3')
    encoder_outputs3, state_h3, state_c3 = encoder_lstm3(encoder_outputs2)

encoder_states = [state_h1, state_c1,
                  state_h2, state_c2,
                  state_h3, state_c3]

decoder_inputs = layers.Input(shape=(max_target_len, ))
decoder_embedding_layer = layers.Embedding(input_dim=target_vocab_size, 
                                           output_dim=embedding_size)
decoder_embedded = decoder_embedding_layer(decoder_inputs)

decoder_lstm1 = layers.LSTM(units=(int(bidirectional) + 1) * 128,
                            return_sequences=True,
                            return_state=True,
                            name='decoder_lstm1')
decoder_outputs1, _, _ = decoder_lstm1(decoder_embedded, initial_state=[state_h1, state_c1])
decoder_lstm2 = layers.LSTM(units=(int(bidirectional) + 1) * 128,
                            return_sequences=True,
                            return_state=True,
                            name='decoder_lstm2')
decoder_outputs2, _, _ = decoder_lstm2(decoder_outputs1, initial_state=[state_h2, state_c2])
decoder_lstm3 = layers.LSTM(units=(int(bidirectional) + 1) * 128,
                            return_sequences=True,
                            return_state=True,
                            name='decoder_lstm3')
decoder_outputs3, _, _ = decoder_lstm3(decoder_outputs2, initial_state=[state_h3, state_c3])

decoder_softmax = layers.Dense(target_vocab_size,
                               activation='softmax')
outputs = decoder_softmax(decoder_outputs3)
    
model = tf.keras.Model([encoder_inputs, decoder_inputs], outputs)
model.summary()

# training
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy')

model.fit(x=[source_input, target_input], y=true_output,
          batch_size=256,
          epochs=5,
          validation_split=0.2)

# prediction model setting
# left part
encoder_model = tf.keras.Model(inputs=encoder_inputs, outputs=encoder_states) 

# right part
decoder_state_input_h1 = layers.Input(shape=((int(bidirectional) + 1) * 128, ))
decoder_state_input_c1 = layers.Input(shape=((int(bidirectional) + 1) * 128, ))
decoder_state_inputs1 = [decoder_state_input_h1, decoder_state_input_c1]

decoder_state_input_h2 = layers.Input(shape=((int(bidirectional) + 1) * 128, ))
decoder_state_input_c2 = layers.Input(shape=((int(bidirectional) + 1) * 128, ))
decoder_state_inputs2 = [decoder_state_input_h2, decoder_state_input_c2]

decoder_state_input_h3 = layers.Input(shape=((int(bidirectional) + 1) * 128, ))
decoder_state_input_c3 = layers.Input(shape=((int(bidirectional) + 1) * 128, ))
decoder_state_inputs3 = [decoder_state_input_h3, decoder_state_input_c3]

decoder_state_inputs = [decoder_state_input_h1, decoder_state_input_c1,
                        decoder_state_input_h2, decoder_state_input_c2,
                        decoder_state_input_h3, decoder_state_input_c3]

decoder_embedded = decoder_embedding_layer(decoder_inputs)
decoder_outputs1, state_h1, state_c1 = decoder_lstm1(decoder_embedded, 
                                                     initial_state=decoder_state_inputs1)
decoder_outputs2, state_h2, state_c2 = decoder_lstm2(decoder_outputs1, 
                                                     initial_state=decoder_state_inputs2)
decoder_outputs3, state_h3, state_c3 = decoder_lstm3(decoder_outputs2, 
                                                     initial_state=decoder_state_inputs3)

#decoder_states_output = [state_h1, state_c1, state_h2, state_c2, state_h3, state_c3]
softmax_outputs = decoder_softmax(decoder_outputs3)

#decoder_model = tf.keras.Model(inputs=[decoder_inputs] + decoder_state_inputs,
#                               outputs=[softmax_outputs] + decoder_states_output)

decoder_model = tf.keras.Model(inputs=[decoder_inputs] + decoder_state_inputs,
                               outputs=[softmax_outputs])
decoder_model.summary()

#%%
# test parallel corpus
s2g_test = pd.read_csv(default_path + '/data/s2g_test_{}.csv'.format(version_num),
                       encoding='cp949',
                       sep=',')
s2g_test = s2g_test.Q
pprint(s2g_test[:10])

#%%
max_length = max([len(x) for x in source_vocab.keys()])
'''
tokenizing이 되지 않은 sequence data에 대한 tokenizing
-> using only vocabulary
'''
def build_test_sequence(sequence, using_unk=False):
    sequence = sequence.replace(' ', '')
    
    # 1. get subword candidates
    candidates = []
    n = len(sequence)
    for start in range(n):
        for end in range(start+1, min(n, start+max_length)+1):
            subword = sequence[start:end]
            if subword in source_vocab:
                candidates.append((subword, start, end, end-start, source_vocab.get(subword)))
            else:
                continue
    
    # 2. get longest matching sequence
    matched = []
    segments = sorted(candidates, key=lambda x:(-x[3], x[1]))
    while segments:
        string, start, end, length, freq = segments.pop(0)
        matched.append((string, start, end, length, freq))
        removals = []
        for i, (_, s, e, _, _) in enumerate(segments):
            if not (e <= start or end <= s): 
                removals.append(i)
        for i in reversed(removals):
            del segments[i]
    
    sorted_matched = sorted(matched, key=lambda x: x[1])
    
    # 3. insert '<UNK>' (optional)
    if using_unk:
        unk_idx = []
        for i in range(len(sorted_matched)):
            if i:
                if sorted_matched[i-1][2] != sorted_matched[i][1]:
                    unk_idx.append(i)
    
        for i in range(len(unk_idx)):
            sorted_matched.insert(unk_idx[i] + i, ['<UNK>'])
        sequence_subwords = [x[0] for x in sorted_matched]
    else:
        sequence_subwords = [x[0] for x in sorted_matched]
    
    # 4. get indexes for subwords and padding
    sequence_idx = [source_vocab.get(x) for x in sequence_subwords] 
    
    result = preprocessing.sequence.pad_sequences([sequence_idx],
                                                  maxlen=max_source_len,
                                                  padding='post')
    return result

#%%
'''
load trained model and build inference model -> layer name 관리 필요
'''    
#del model
loaded_model = load_model(default_path + '/model/model_{}.h5'.format(version_num))
loaded_model.summary()

# load layers
loaded_encoder_inputs = loaded_model.get_layer('input_1')
loaded_encoder_embedding_layer = loaded_model.get_layer('embedding')
loaded_encoder_lstm1 = loaded_model.get_layer('bidirectional')
loaded_encoder_lstm2 = loaded_model.get_layer('bidirectional_1')
loaded_encoder_lstm3 = loaded_model.get_layer('bidirectional_2')

loaded_decoder_inputs = loaded_model.get_layer('input_2')
loaded_decoder_embedding_layer = loaded_model.get_layer('embedding_1')
loaded_decoder_lstm1 = loaded_model.get_layer('decoder_lstm1')
loaded_decoder_lstm2 = loaded_model.get_layer('decoder_lstm2')
loaded_decoder_lstm3 = loaded_model.get_layer('decoder_lstm3')
loaded_dense_softmax = loaded_model.get_layer('dense')

# build encoder & decoder model
# 1. encoder
loaded_encoder_embedded = loaded_encoder_embedding_layer(loaded_encoder_inputs)
if bidirectional:
    encoder_outputs1, forward_h1, forward_c1, backward_h1, backward_c1 = loaded_encoder_lstm1(loaded_encoder_embedded).output
    encoder_outputs2, forward_h2, forward_c2, backward_h2, backward_c2 = loaded_encoder_lstm2(encoder_outputs1).output
    encoder_outputs3, forward_h3, forward_c3, backward_h3, backward_c3 = loaded_encoder_lstm3(encoder_outputs2).output
    state_h1 = layers.Concatenate()([forward_h1, backward_h1])
    state_c1 = layers.Concatenate()([forward_c1, backward_c1])
    state_h2 = layers.Concatenate()([forward_h2, backward_h2])
    state_c2 = layers.Concatenate()([forward_c2, backward_c2])
    state_h3 = layers.Concatenate()([forward_h3, backward_h3])
    state_c3 = layers.Concatenate()([forward_c3, backward_c3])

else:
    encoder_outputs1, state_h1, state_c1 = loaded_encoder_lstm1(loaded_encoder_embedded).output
    encoder_outputs2, state_h2, state_c2 = loaded_encoder_lstm2(encoder_outputs1).output
    encoder_outputs3, state_h3, state_c3 = loaded_encoder_lstm3(encoder_outputs2).output

encoder_states = [state_h1, state_c1,
                  state_h2, state_c2,
                  state_h3, state_c3]

loaded_encoder_model = tf.keras.Model(inputs=loaded_encoder_inputs, outputs=encoder_states) 
loaded_encoder_model.summary()

# 2. decoder
decoder_state_input_h1 = layers.Input(shape=((int(bidirectional) + 1) * 128, ))
decoder_state_input_c1 = layers.Input(shape=((int(bidirectional) + 1) * 128, ))
decoder_state_inputs1 = [decoder_state_input_h1, decoder_state_input_c1]

decoder_state_input_h2 = layers.Input(shape=((int(bidirectional) + 1) * 128, ))
decoder_state_input_c2 = layers.Input(shape=((int(bidirectional) + 1) * 128, ))
decoder_state_inputs2 = [decoder_state_input_h2, decoder_state_input_c2]

decoder_state_input_h3 = layers.Input(shape=((int(bidirectional) + 1) * 128, ))
decoder_state_input_c3 = layers.Input(shape=((int(bidirectional) + 1) * 128, ))
decoder_state_inputs3 = [decoder_state_input_h3, decoder_state_input_c3]

decoder_state_inputs = [decoder_state_input_h1, decoder_state_input_c1,
                        decoder_state_input_h2, decoder_state_input_c2,
                        decoder_state_input_h3, decoder_state_input_c3]

#decoder_state_inputs = [decoder_state_inputs1,
#                        decoder_state_inputs2,
#                        decoder_state_inputs3]

loaded_decoder_embedded = loaded_decoder_embedding_layer(loaded_decoder_inputs)

decoder_outputs1, state_h1, state_c1 = loaded_decoder_lstm1(loaded_decoder_embedded, 
                                                            initial_state=decoder_state_inputs1)
decoder_outputs2, state_h2, state_c2 = loaded_decoder_lstm2(decoder_outputs1, 
                                                            initial_state=decoder_state_inputs2)
decoder_outputs3, state_h3, state_c3 = loaded_decoder_lstm3(decoder_outputs2, 
                                                            initial_state=decoder_state_inputs3)

softmax_outputs = loaded_dense_softmax(decoder_outputs3)

loaded_decoder_model = tf.keras.Model(inputs=[loaded_decoder_inputs] + decoder_state_inputs,
                                      outputs=[softmax_outputs])
loaded_decoder_model.summary()

#%%
# encoder & decoder model saving
with open(default_path + '/model/encoder_model_{}.json'.format(version_num), 'w', encoding='utf8') as f:
    f.write(encoder_model.to_json())
encoder_model.save_weights(default_path + '/model/encoder_model_weights_{}.h5'.format(version_num))

with open(default_path + '/model/decoder_model_{}.json'.format(version_num), 'w', encoding='utf8') as f:
    f.write(decoder_model.to_json())
decoder_model.save_weights(default_path + '/model/decoder_model_weights_{}.h5'.format(version_num))

#%%
def load_full_model(model_filename, model_weights_filename):
    with open(model_filename, 'r', encoding='utf8') as f:
        model = model_from_json(f.read())
    model.load_weights(model_weights_filename)
    return model

loaded_encoder_model = load_full_model(default_path + '/model/encoder_model_{}.json'.format(version_num), 
                                       default_path + '/model/encoder_model_weights_{}.h5'.format(version_num))
loaded_decoder_model = load_full_model(default_path + '/model/decoder_model_{}.json'.format(version_num),
                                       default_path + '/model/decoder_model_weights_{}.h5'.format(version_num))
    
#%%
def decode_sequence(sequence):
    input_seq = np.array(sequence).reshape((1, max_source_len))
    
    # encoder의 마지막 hidden state 값을 저장
    last_hidden_states = loaded_encoder_model.predict(input_seq)
    # <sos>에 해당하는 input 생성
    target_seq = np.zeros((1, max_target_len))
    target_seq[0, 0] = target_vocab.get('<sos>')
    
    stop_condition = False
    decoded_sequence = []
    idx = 0
    while not stop_condition:
#        output_subwords, _, _ = loaded_decoder_model.predict([target_seq] + last_hidden_states)
        output_subwords = loaded_decoder_model.predict([target_seq] + last_hidden_states)
        max_vocab_index = np.argmax(output_subwords[0][idx])
        argmax_subword = target_idx2word.get(max_vocab_index)
        decoded_sequence.append(argmax_subword)
        
        if argmax_subword == '<eos>' or len(decoded_sequence) >= max_target_len:
            stop_condition = True
        else:
            idx += 1
            target_seq[0, idx] = max_vocab_index
    
    return decoded_sequence

#%%
# decode test corpus
s2g_test_decoded = []
for i in progressbar(range(len(s2g_test))):
    decoded = decode_sequence(build_test_sequence(s2g_test[i], using_unk=True))
    s2g_test_decoded.append(' '.join(decoded).replace(' <eos>', ''))
pprint(s2g_test_decoded[:10])

decode_sequence(build_test_sequence(s2g_test[0], using_unk=True))

# test 결과 저장
model_format = '{}_{}'.format(''.join(['deep' if deep else 'shallow']), ''.join(['bi' if bidirectional else 'uni']))
final_save = pd.concat([pd.DataFrame({'Q': s2g_test}), pd.DataFrame({'C': s2g_test_decoded})], axis=1)
final_save.to_csv(default_path + '/result/s2g_{}_result_{}.csv'.format(model_format, version_num),
                  sep=',',
                  index=False,
                  header=True,
                  encoding='cp949')

#%%
# on-the-fly test
on_the_fly_test = '그 뭐냐 외국 여자가 용산구에 얼마나 있는지 그래프 그려줘'

print(' '.join([source_idx2word.get(x) for x in build_test_sequence(on_the_fly_test, using_unk=True)[0]]))
print(' '.join([source_idx2word.get(x) for x in build_test_sequence(on_the_fly_test, using_unk=False)[0]]))

print(' '.join(decode_sequence(build_test_sequence(on_the_fly_test, using_unk=True))).replace(' <eos>', ''))
print(' '.join(decode_sequence(build_test_sequence(on_the_fly_test, using_unk=False))).replace(' <eos>', ''))

#%%























