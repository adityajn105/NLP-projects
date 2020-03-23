"""
Named Entity Recognition

By - Aditya Jain
"""
import numpy as np
import pandas as pd
import time
import pickle as pkl

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, Embedding, TimeDistributed, LSTM, Input, Bidirectional, Dropout
from tensorflow.keras.models import Model, save_model

tokens = {'<pad>': 0}
embedding_matrix = [ [0]*50 ]
i = 1
#look for this file in datasets
with open('glove.6B.50d.txt', "r") as fp:
  for line in fp.readlines():
    a = line.split()
    tokens[ a[0].lower() ] = i
    embedding_matrix.append( list(map(float, a[1:])) )
    i+=1
embedding_matrix = np.array(embedding_matrix)
VOCAB_SIZE = len(tokens)
UNK_IDX = tokens['unk'] #index of unknown token

pkl.dump( tokens, open("tokens.pkl","wb")) )

#read ner_dataset from datasets directory
ner = pd.read_csv('../datasets/ner_dataset.csv', encoding="ISO-8859-1", error_bad_lines=False, engine='c')
ner['Sentence #'].fillna(method="ffill", inplace=True)
ner.dropna(inplace=True)

NER_SIZE = ner.Tag.nunique() + 1
print("Output Size", NER_SIZE, "Total Words", VOCAB_SIZE)

#create idx to output, output to idx dictionaries
NER_DICT, REVERSE_NER_DICT = {}, {}
idx = 0
for tag in ner.Tag.unique():
    NER_DICT[tag] = idx
    REVERSE_NER_DICT[idx] = tag
    idx+=1

#apply indexing to word and output
ner['Word'] = ner['Word'].apply( lambda x: tokens.get( x.lower(), UNK_IDX) )
ner['Tag'] = ner['Tag'].apply(lambda x: NER_DICT[x])
ner.head()

#prepare input sequence and target sequence
sentences = ner.groupby('Sentence #').Word.apply(lambda x: np.array(x)).values
targets = ner.groupby('Sentence #').Tag.apply(lambda x: np.array(x)).values
sentences.shape, targets.shape

#We are covering 98% of sentences, take percentile for validation
MAX_SEQ_LEN = 40

#apply padding to input and output sequences
sentences = pad_sequences(sentences, maxlen=MAX_SEQ_LEN, padding="post", value=0)
targets = pad_sequences(targets, maxlen=MAX_SEQ_LEN, padding="post", value=0)
sentences.shape, targets.shape

#one Hot encode target sequence
targets_final = np.zeros( ( *targets.shape, NER_SIZE) )
for i in range(TOTAL_SIZE):
  for j in range(MAX_SEQ_LEN):
    targets_final[ i, j, targets[i,j] ] = 1
targets_final.shape


TOTAL_SIZE = len(sentences)
TEST_SIZE = int(0.2*TOTAL_SIZE)

#train test split
trainX, valX, trainY, valY = train_test_split( sentences, targets_final, test_size=0.2,  shuffle=True )

#prepare model for generating embedding
inp = Input( shape=( MAX_SEQ_LEN,) )
emb = Embedding( input_dim=VOCAB_SIZE, output_dim=50, weights=[embedding_matrix], 
                trainable=False, input_length=MAX_SEQ_LEN)(inp)
embedding = Model( inp, emb )
save_model( embedding, "embedding_model.h5" )


#Model for identifying tags of each word
inp = Input( shape=(MAX_SEQ_LEN, 50) )
drop = Dropout(0.1)(inp)
#two bidirectinal LSTM layers
lstm1 = LSTM( 30, return_sequences=True, recurrent_dropout=0.1)
seq1 = Bidirectional(lstm1)( drop )
lstm2 = LSTM( 30, return_sequences=True, recurrent_dropout=0.1)
seq2 = Bidirectional(lstm2)( seq1 )
# TIME_DISTRIBUTED -> ( MAX_SEQ_LEN, 50 ) -> (MAX_SEQ_LEN, NER_SIZE)
tags = TimeDistributed( Dense(NER_SIZE, activation="relu") )(seq2)
model = Model( inp, tags )
model.compile( optimizer="rmsprop", loss="categorical_crossentropy", metrics=[ "accuracy" ] )

#batch generator for model training
def getBatch(sentences, targets, batch_size=128):
  n = len(sentences)//batch_size
  for i in range( n+1 ):
    x = sentences[ i*batch_size : (i+1)*batch_size ]
    x = embedding.predict(x)
    y = targets[ i*batch_size : (i+1)*batch_size ]
    yield x,y

#do training
all_metrics = []
for epoch in range(1,51):
  loss, acc, i, t0 = 0,0,0,time.time()
  #get batches
  for x,y in getBatch(trainX, trainY, batch_size=128):
    metrics_train = model.train_on_batch(x, y)
    loss, acc, i = loss+metrics_train[0], acc+metrics_train[1], i+1
  metrics_val = model.test_on_batch( embedding.predict(valX), valY )
  all_metrics.append( ( loss/i, acc/i, metrics_val[0], metrics_val[1]  ) )
  print(f" Epoch {epoch:2.0f} - {(time.time()-t0):.0f}s - loss: {loss/i:.4f} - accuracy: {acc/i:.4f} - val_loss: {metrics_val[0]:.4f} - val_accuracy: {metrics_val[1]:.4f}")
save_model( model, "ner_tagging_model.h5" )

#plot curves
epochs = len(all_metrics)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15,5))
ax[0].plot( range(epochs), list(map(lambda x: x[0], all_metrics)), label="Train Loss" )
ax[0].plot( range(epochs), list(map(lambda x: x[2], all_metrics)), label="Val Loss" )
ax[0].legend();ax[0].set_xlabel('Epochs');ax[0].set_ylabel('Loss');ax[0].grid()
ax[0].set_title("NER Model Training and Validation Loss curves")

ax[1].plot( range(epochs), list(map(lambda x: x[1], all_metrics)), label="Train Accuracy" )
ax[1].plot( range(epochs), list(map(lambda x: x[3], all_metrics)), label="Val Accuracy" )
ax[1].legend();ax[1].set_xlabel('Epochs');ax[1].set_ylabel('Accuracy');;ax[1].grid()
ax[1].set_title("NER Model Training and Validation Accuracy Curves")

def getTags(sentence):
  inp = []
  for word in sentence.split():
    inp.append( tokens.get( word.lower(), UNK_IDX) )
  ln = len(inp)
  inp = pad_sequences( np.expand_dims(inp,axis=0), maxlen=MAX_SEQ_LEN, padding="post", value=0 )
  emb = embedding.predict(inp)
  ans = model.predict( emb )[0]
  ans = [ np.argmax(tag) for tag in ans ][:ln]
  return " ".join([ REVERSE_NER_DICT[pos] for pos in ans ])



if __name__=='__main__':
    while True:
        sentence = input("Enter Sentence")
        print(getTags( sentence ))
        if input("Continue? (y/n)")=='n':
            break