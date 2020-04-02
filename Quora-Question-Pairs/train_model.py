import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import re
import os

from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.layers import Input, Embedding, LSTM, Lambda, Bidirectional, Dropout
from tensorflow.keras.layers import GlobalMaxPool1D, GlobalAvgPool1D, Concatenate, Dense
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.metrics import AUC

from tensorflow.keras.callbacks import ModelCheckpoint

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
train = train.dropna(how="any")
print(train.shape,test.shape)

def preprocess(ques):
  ques = ques.strip()
  ques = re.sub( "\n", " ", ques)
  ques = re.sub( "'s|’s", "", ques)
  ques = re.sub( "s' ", "s ", ques)
  #take only alphabetical chars
  preprocessed_ques = "".join([ c.lower() for c in ques if c.isalpha() or c==" " ])
  #remove some stop words
  stop_words = [ 'the', 'a', 'an', 'some' ]
  preprocessed_ques = filter( lambda x: x not in stop_words and len(x)!=1 and len(x)<=20, preprocessed_ques.split() )
  return " ".join(preprocessed_ques).strip()
train.question1 = train.question1.apply(preprocess)
train.question2 = train.question2.apply(preprocess)
print(train.head(2))

#generate glove embedding matric and dictionary
EMBEDDING_DIM = 100
glove_dict = { '<pad>' : 0 }
embedding_matrix = [ [0]*EMBEDDING_DIM ]
idx = 1
with open(f'glove.6B.{EMBEDDING_DIM}d.txt',"r") as fp:
  for line in fp.readlines():
    line = line.split()
    glove_dict[ line[0] ] = idx
    embedding_matrix.append( list(map(float, line[1:])) )
    idx+=1
embedding_matrix = np.array(embedding_matrix)
print("embedding matrix shape :", embedding_matrix.shape, 'length of glove dict :', len(glove_dict))

def encodeQues( ques ):
  encoded = []
  for word in ques.split():
    encoded.append( glove_dict.get( word, glove_dict['unk'] ) )
  return encoded

train.question1 = train.question1.apply(encodeQues)
train.question2 = train.question2.apply(encodeQues)
print(train.head(2))

lens = list(map(len,train.question1.values))+list(map(len,train.question2.values))
sns.distplot(lens)
plt.show()

MAX_SEQ_LEN = 35
question1 = pad_sequences( train.question1.values, maxlen= MAX_SEQ_LEN, padding='post', value=0 )
question2 = pad_sequences( train.question2.values, maxlen= MAX_SEQ_LEN, padding='post', value=0 )
y = train.is_duplicate.values


# Model variables
left_input = Input( shape=(MAX_SEQ_LEN,) )
right_input = Input( shape=(MAX_SEQ_LEN,) )

embedding_layer = Embedding(len(embedding_matrix), EMBEDDING_DIM, 
                            weights=[embedding_matrix],
                            trainable=False)

# Embedded version of the inputs
encoded_left = embedding_layer(left_input)
encoded_right = embedding_layer(right_input)

# Since this is a siamese network, both sides share the same LSTM
shared_bilstm = Bidirectional( LSTM(100, return_sequences=True, dropout=0.1, recurrent_dropout=0.1) )
left_output = shared_bilstm(encoded_left)
right_output = shared_bilstm(encoded_right)

maxpool = GlobalMaxPool1D()
avgpool = GlobalAvgPool1D()
concatenate1 = Concatenate()
dropout1 = Dropout(0.1)
x1 = concatenate1([ maxpool(left_output), avgpool(left_output)  ])
x2 = concatenate1([ maxpool(right_output), avgpool(right_output) ])
x1 = dropout1(x1)
x2 = dropout1(x2)

sqr_diff = Lambda( lambda tensors: K.pow((K.square(tensors[0])-K.square(tensors[1])), 0.5), name="Squared_diff" )
abs_diff = Lambda( lambda tensors : K.abs(tensors[0]-tensors[1]), name="Absolute_diff" )
concatenate2 = Concatenate()
diff = concatenate2([ sqr_diff([x1,x2]), abs_diff([x1,x2]) ])
diff = Dropout(0.1)(diff)

diff = Dense(100, activation="relu")(diff)
diff = Dropout(0.1)(diff)
out = Dense(1, activation="sigmoid")(diff)

model = Model( [left_input, right_input], out )
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=[AUC(name="auc")])
print(model.summary())

if os.path.isfile('model_best_checkpoint.h5'):
  model.load_weights('model_best_checkpoint.h5')
model_checkpoint  = ModelCheckpoint('model_best_checkpoint.h5', save_best_only=True, save_weights_only=True, 
                                    monitor='val_loss', mode='min', verbose=1)
callback_list= [model_checkpoint]
BATCH_SIZE = 256
history = model.fit([question1, question2], y, batch_size=BATCH_SIZE, 
                     epochs=20, validation_split=0.2, callbacks=callback_list)

