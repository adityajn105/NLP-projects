import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd
import re
import time
from math import ceil

from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Embedding, Bidirectional, Dense, LSTM, Dropout, Conv1D
from tensorflow.keras.layers import GlobalMaxPool1D, GlobalAvgPool1D, Concatenate, SpatialDropout1D
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import AUC

data = pd.read_csv('../datasets/train.csv')
data.drop('id',axis=1,inplace=True)
print(data.shape)

def preprocess_comment( comment ):
  comment = comment.strip()
  comment = re.sub( "\n", " ", comment)
  comment = re.sub( "'s|’s", "", comment)
  comment = re.sub( "s' ", "s ", comment)
  #take only alphabetical chars
  preprocessed_comment = "".join([ c.lower() for c in comment if c.isalpha() or c==" " ])
  #remove some stop words
  stop_words = [ 'the', 'a', 'an', 'some', 'once' ]
  preprocessed_comment = filter( lambda x: x not in stop_words and len(x)!=1 and len(x)<=20, preprocessed_comment.split() )
  return " ".join(preprocessed_comment).strip()
data.comment_text = data.comment_text.apply( preprocess_comment )
print( data.head() )

#generate glove embedding matric and dictionary
EMBEDDING_DIM = 100
glove_dict = { '<pad>' : 0 }
embedding_matrix = [ [0]*EMBEDDING_DIM ]
idx = 1
with open(f'../datasets/glove.6B.{EMBEDDING_DIM}d.txt',"r") as fp:
  for line in fp.readlines():
    line = line.split()
    glove_dict[ line[0] ] = idx
    embedding_matrix.append( list(map(float, line[1:])) )
    idx+=1
embedding_matrix = np.array(embedding_matrix)
print("embedding matrix shape :", embedding_matrix.shape, 'length of glove dict :', len(glove_dict))

targets = data.iloc[ : , 1:].values
inputs = data.comment_text.apply(  lambda comment : [ glove_dict.get(x, glove_dict['unk']) for x in comment.split() ])

sns.distplot( list(map( len, inputs )) )
plt.show()

MAXIMUM_SEQ_LEN = 200 #We will be covering approx 95 percent sentences length
VOCAB_LENGTH = len(glove_dict) # this is our vocab length
print('MAXIMUM_SEQ_LENGTH: ', MAXIMUM_SEQ_LEN, 'VOCAB_LENGTH: ',VOCAB_LENGTH)

#padding input sequences
inputs = pad_sequences( inputs, maxlen= MAXIMUM_SEQ_LEN, padding='post', value=0 )
print("Input Tensor Shape: ", inputs.shape)

#train test split
Xtrain, Xval, Ytrain, Yval = train_test_split( inputs, targets, test_size=0.3)

def buildModel():
	inp = Input( (MAXIMUM_SEQ_LEN,) )
	#use embeddings
	emb = Embedding( VOCAB_LENGTH, EMBEDDING_DIM, weights = [embedding_matrix], trainable=False )(inp)
	#to drop some embedding instead of particular cells
	emb = SpatialDropout1D(0.2)(emb)
	#generate 100(fwd) + 100(bwd) hidden states 
	hidden_states = Bidirectional( LSTM( 100, return_sequences=True, dropout=0.1, recurrent_dropout=0.1 ) )(emb)
	#on each hidden state use 100*64 kernels of size 3
	conv = Conv1D(64, kernel_size = 3, padding = "valid", kernel_initializer = "glorot_uniform")(hidden_states)
	#take maximum for each cell of all hidden state
	x1 = GlobalMaxPool1D()(conv)
	x2 = GlobalAvgPool1D()(conv)
	#cocatenate both polling
	x = Concatenate()([x1,x2])
	x = Dropout(0.2)(x)
	x = Dense( 50, activation='relu' )(x)
	x = Dropout(0.1)(x)
	out = Dense(6, activation='sigmoid')(x)
	model  = Model( inp, out)

	model.compile( loss="binary_crossentropy", optimizer="adam", metrics=[ AUC(name="auc")] )

model = buildModel()
print(model.summary())

#batch generator
def getBatch( X, y=None, batch_size=128 ):
  num_batches = len(X)//batch_size
  for i in range(num_batches+1):
    if type(y)==type(None): yield X[i*batch_size: (i+1)*batch_size]
    else: yield X[i*batch_size: (i+1)*batch_size], y[i*batch_size: (i+1)*batch_size]

#training
all_metrics = []
PATIENCE, CURRENT_PATIENCE  = 2, 0
BATCH_SIZE=512
BEST_LOSS = float('inf')
for epoch in range(1,21):

  tloss, tauc, ti, t0 = 0,0,0,time.time()
  for x,y in getBatch(Xtrain, Ytrain, BATCH_SIZE):
    metrics_train = model.train_on_batch(x, y)
    tloss, tauc, ti = tloss+metrics_train[0], tauc+metrics_train[1], ti+1
    print(f'Training Progress: {ti}/{ceil(len(Xtrain)/BATCH_SIZE)}', end="\r")

  vloss, vauc, vi = 0, 0, 0
  for x,y in getBatch( Xval, Yval, batch_size = 512 ):
    metrics_val = model.test_on_batch(x, y)
    vloss, vauc, vi = vloss+metrics_val[0], vauc+metrics_val[1], vi+1
    print(f'Validation Progress: {vi}/{ceil(len(Xval)/BATCH_SIZE)}', end="\r")

  all_metrics.append( ( tloss/ti, tauc/ti, vloss/vi, vauc/vi  ) )
  print(f" Epoch {epoch:2.0f} - {(time.time()-t0):.0f}s - loss: {tloss/ti:.4f} - auc: {tauc/ti:.4f} - val_loss: {vloss/vi:.4f} - val_auc: {vauc/vi:.4f}")

  if all_metrics[-1][2] >= BEST_LOSS: CURRENT_PATIENCE += 1
  else: BEST_LOSS = all_metrics[-1][2]; CURRENT_PATIENCE = 0
  if CURRENT_PATIENCE == PATIENCE: print("Early Stopping!!");break;


#plot curves
epochs = len(all_metrics)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15,5))
ax[0].plot( range(epochs), list(map(lambda x: x[0], all_metrics)), label="Train Loss" )
ax[0].plot( range(epochs), list(map(lambda x: x[2], all_metrics)), label="Val Loss" )
ax[0].legend();ax[0].set_xlabel('Epochs');ax[0].set_ylabel('Loss');ax[0].grid()
ax[0].set_title("Model training and Validation Loss curves")

ax[1].plot( range(epochs), list(map(lambda x: x[1], all_metrics)), label="Train AUC" )
ax[1].plot( range(epochs), list(map(lambda x: x[3], all_metrics)), label="Val AUC" )
ax[1].legend();ax[1].set_xlabel('Epochs');ax[1].set_ylabel('Accuracy');ax[1].grid()
ax[1].set_title("Model Training and Validation AUC Curves")
plt.show()

#prepare submission data
submission_comments = pd.read_csv('test.csv').comment_text
submission = pd.read_csv('sample_submission.csv')
submission_comments = submission_comments.apply( preprocess_comment )
submission_comments = submission_comments.apply(  
    lambda comment : [ glove_dict.get(x, glove_dict['unk']) for x in comment.split() ])
submission_comments = pad_sequences( submission_comments, maxlen= MAXIMUM_SEQ_LEN, padding='post', value=0 )

#generate predictions for submissions.
ans,i = np.empty( (0,6) ),1
for x in getBatch(submission_comments, batch_size=2048):
  ans = np.append(ans,model.predict(x),axis=0)
  print(f"Progress: {i}/{int(len(submission_comments)//2048)}",end="\r");i+=1
print(ans.shape)

#write submissions
submission.iloc[ :, 1:7 ] = ans
submission.to_csv("submission.csv",index=False)