import numpy as np
import pandas as pd

%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM
from tensorflow.keras.layers import RepeatVector, Concatenate, Dense, Dot
from tensorflow.keras.layers import Lambda

from tensorflow.keras import backend as K

from tensorflow.keras.models import Model

#get data from here - http://www.manythings.org/anki/spa-eng.zip
#also you can try multiple other translations
data = pd.read_csv( 'spa.txt', sep='\t', header=None )
NUM_SAMPLES = 120000
data = data.iloc[:NUM_SAMPLES,:2]
data.columns = [ 'input', 'target' ]
print(data.shape)

def preprocess_input(x):
  to_replace = { "'s": " 's", "n't": " ot", "'ll": " will", "-": " ", "'ve": " have", "'re": " are",
                "'d": " would", "'m": " am", ".":" .", "?":" ?", ",":"", "!":" !" }
  for k,v in to_replace.items():
    x = x.replace( k, v )
  return x.lower()

def preprocess_target(x):
  to_replace = {".":" .", "?":" ?", ",":"", "!":" !" }
  for k,v in to_replace.items():
    x = x.replace(k,v)
    return x.lower()
data.target = data['target'].apply(preprocess_target)
data.input = data['input'].apply(preprocess_input)


## Preparing GLove word embedding and embedding_matrix
word2idx_encoder = {'<pad>': 0}
idx2word_encoder = {0:'<pad>'}
embedding_matrix = [ [0]*50 ]
i = 1
#look for this file in datasets
with open('glove.6B.50d.txt', "r") as fp:
  for line in fp.readlines():
    a = line.split()
    word2idx_encoder[ a[0].lower() ] = i
    idx2word_encoder[ i ] = a[0].lower()
    embedding_matrix.append( list(map(float, a[1:])) )
    i+=1
embedding_matrix = np.array(embedding_matrix)
ENCODER_VOCAB_SIZE = len(word2idx_encoder)
UNK_IDX = word2idx_encoder['unk'] #index of unknown token
ENCODER_EMBEDDING_DIM = 50


encoder_input = []
unk=set()
for sentence in data.input.values:
  sequence = []
  for word in sentence.split():
    if word not in word2idx_encoder: unk.add(word)
    sequence.append( word2idx_encoder.get( word, UNK_IDX ) )
  encoder_input.append(sequence)
print("Length of encoder input: ",len(encoder_input),"total unknowns in input vocab ",len(unk))


word2idx_decoder = { '<pad>':0, '<sos>':1, '<eos>':2 }
idx2word_decoder = { 0:'<pad>', 1:'<sos>', 2:'<eos>' }
idx = 3
for sentence in data.target.values:
  for word in sentence.split():
    if word in word2idx_decoder: continue
    else:
      word2idx_decoder[word] = idx
      idx2word_decoder[idx] = word
      idx+=1

DECODER_VOCAB_SIZE = len(word2idx_decoder)
print("decoder vocab size: ",DECODER_VOCAB_SIZE)


decoder_input, decoder_output = [], []
for sentence in data.target.values:
  sequence1, sequence2 = [ 1 ], []
  for word in sentence.split():
    sequence1.append( word2idx_decoder[word] )
    sequence2.append( word2idx_decoder[word] )
  sequence2.append( 2 )
  decoder_input.append(sequence1)
  decoder_output.append(sequence2)

MAX_ENCODER_SIZE = Tx = max( len(x) for x in encoder_input )
MAX_DECODER_SIZE = Ty = max( len(x) for x in decoder_input )
print("MAX_ENCODER_SIZE: ", MAX_ENCODER_SIZE, "MAX_DECODER_SIZE", MAX_DECODER_SIZE )


encoder_input = pad_sequences( encoder_input, maxlen=MAX_ENCODER_SIZE, padding='post', value=0 )
decoder_input = pad_sequences( decoder_input, maxlen=MAX_DECODER_SIZE, padding="post", value=0 )
decoder_output = pad_sequences( decoder_output, maxlen=MAX_DECODER_SIZE, padding="post", value=0 )
print( "encoder input shape", encoder_input.shape, "decoder input shape", decoder_input.shape  )


#Encoder Part
#pass all input through embedding layer, to get embeddings
#pass all embeddings to Bi-LSTM to get all sequences of hidden states (h1...htx)
LATENT_DIM_EN = 50 #M1
LATENT_DIM_DE = 60 #M2

encoder_inp = Input( shape=(MAX_ENCODER_SIZE,) ) #(_,Tx)

encoder_embedding = Embedding( ENCODER_VOCAB_SIZE, ENCODER_EMBEDDING_DIM, weights=[embedding_matrix], trainable=False )
embeddings_en = encoder_embedding( encoder_inp ) #(_,Tx, ENCODER_EMBEDDING_DIM)

encoder_bilstm = Bidirectional( LSTM( LATENT_DIM_EN, return_sequences=True, recurrent_dropout=0.1 ) )
hidden_states = encoder_bilstm( embeddings_en ) #(_,Tx, 2*M1)

#Attention Part
#Repear s(t-1) using repeae vector
#concatenate s(t-1) with each hidden state h_t
#Pass it though a neurel network with output of one neuron
#apply softmax over time axis, other wise it alphas will be one
#get weigher hidden states (when we multiple alpha with hidden state)
#sum all weighted hidden state this is context
#last 2 steps can be achieved by dot product over axis=1
def softmax_over_time(x): #(softmax on time axis (axis=1) instead of axis=-1)
  e = K.exp( x - K.max(x, axis=1, keepdims=True) )
  s = K.sum(e, axis=1, keepdims=True)
  return e/s

attn_repeatvector = RepeatVector(Tx) #to repeat previous decoder-state s(t-1) over Tx times
attn_concatenate = Concatenate(axis=-1) #to concatenate s(t-1) with every encoder hidden_state
attn_dense = Dense( 10, activation="tanh" ) #a dense layer
attn_alpha = Dense( 1, activation=softmax_over_time ) #to get importance of each hidden state
attn_context = Dot(axes=1) # weighted(importnace) sum of all hidden states = \sum (h_i.alpha_i)

def one_step_attention( h_states, s_prev ):
  x = attn_repeatvector( s_prev ) #(_,Tx, M2)
  x = attn_concatenate( [ h_states, x ] ) #(_,Tx, 2*M1+M2)
  x = attn_dense(x) #(_,Tx, 10)
  alpha = attn_alpha(x) #(_,Tx, 1)
  context = attn_context([alpha, h_states]) #(_,1,2*M1)
  return context, alpha


#Decoder Part
#take embedding of all decoder input which is actually output one step behind (for teacher forcing)
#for each output time step t,
#	generate context
#	concat s(t-1) with Y(t-1)
#	pass this through dense layer
#	pass this thorugh decoder LSTM, get output and replace hidden state, and cell state with older one
#	pass output to dense layer with softmax activation to get next word probabilities.
# 	append this to a list
#output list is of shape ( Ty, None, VOCAB_SIZE ), change it to (Nonde, Ty, VOCAB_SIZE) using lambda layer
decoder_inp = Input( shape=(MAX_DECODER_SIZE,) ) #(_,Ty)
initial_decoder_h = Input( shape=( LATENT_DIM_DE, ) ) #(_,M2)
initial_decoder_c = Input( shape=( LATENT_DIM_DE, ) ) #(_,M2)

decoder_embedding = Embedding( DECODER_VOCAB_SIZE, 50 )
embeddings_de = decoder_embedding(decoder_inp) #_,Ty, 50

concat_context_word_prev = Concatenate(axis=-1, name="word_context_concat")
decoder_lstm = LSTM( LATENT_DIM_DE, return_state=True, name="decoder_lstm", recurrent_dropout=0.1 )
dense_context = Dense(100, activation='tanh', name="decoder_context")
dense_decoder = Dense( DECODER_VOCAB_SIZE, activation='softmax', name="decoder_dense") 

s = initial_decoder_h #(_,M2)
c = initial_decoder_c #(_,M2)
outputs = [] #to save each decoding timestep output
for t in range(Ty):
  context, _ = one_step_attention( hidden_states, s ) #(_,1, 2*M1)

  selector = Lambda( lambda x: x[:,t:t+1], name=f"selector_{t}" ) #for teacher forcing
  word_embedding = selector(embeddings_de ) #(_,1, 50)
  context_word = concat_context_word_prev([context,word_embedding]) #(_,1, 2*M1+50)
  context_word = dense_context( context_word ) #(_,1, 100)

  output, s, c = decoder_lstm( context_word, initial_state=[s,c] ) #(_,1,M2), (_,M2), (_,M2)
  output = dense_decoder(output) #(_, DECODER_VOCAB_SIZE)
  outputs.append(output) # after loop it will be a list of length Ty (_ , DEOCDER_VOCAB_SIZE)

# to change outputs shape to (_, Ty, DEOCDER_VOCAB_SIZE)
def stack_and_transpose(x):
  x = K.stack(x) # it will convert list to a tensor of ( Ty, _, DEOCDER_VOCAB_SIZE )
  x = K.permute_dimensions( x, pattern=(1,0,2) ) #(_, Ty, DEOCDER_VOCAB_SIZE)
  return x
stacker = Lambda(stack_and_transpose, name="stack_and_transpose")
outputs = stacker(outputs) #(_, Ty, DEOCDER_VOCAB_SIZE)



#custom acc because we dont want to consider padding
def acc(y_true, y_pred):
  # both are of shape ( _, Ty, DEOCDER_VOCAB_SIZE )
  targ = K.argmax(y_true, axis=-1)
  pred = K.argmax(y_pred, axis=-1)
  correct = K.cast(  K.equal(targ,pred), dtype='float32') #cast bool tensor to float

  # 0 is padding, don't include those- mask is tensor representing non-pad value
  mask = K.cast(K.greater(targ, 0), dtype='float32') #cast bool-tensor to float 
  n_correct = K.sum(mask * correct) #
  n_total = K.sum(mask)
  return n_correct / n_total

#custom loss because we dont want to consider padding
def loss(y_true, y_pred):
   # both are of shape ( _, Ty, DEOCDER_VOCAB_SIZE )
  mask = K.cast(y_true > 0, dtype='float32')
  out = mask * y_true * K.log(y_pred) #cross entopy loss
  return -K.sum(out) / K.sum(mask)

model =  Model( [ encoder_inp, decoder_inp, initial_decoder_h, initial_decoder_c ], outputs )
model.compile( optimizer="rmsprop", loss=loss, metrics=[acc])
print(model.summary())


#to convert a batch of decoder outputs to one hot.
def ohe_decoder_output( x ):
  ohe = np.zeros( (*(x.shape), DECODER_VOCAB_SIZE) )
  for i in range(len(x)):
    for j in range(MAX_DECODER_SIZE):
      ohe[i,j,x[i][j]] = 1
  return ohe
  
#Using generator due to huge volume of data.
def generator( encoder_inp, decoder_inp, decoder_out, bs=128 ):
  total_batches = len(encoder_inp)//bs
  initial_h = np.zeros( (bs, LATENT_DIM_DE) )
  initial_c = np.zeros( (bs, LATENT_DIM_DE) )
  for i in range(total_batches):
    yield [ encoder_inp[i*bs:(i+1)*bs], decoder_inp[i*bs:(i+1)*bs], initial_h, initial_c], ohe_decoder_output( decoder_out[i*bs:(i+1)*bs] )

train_en_inp, val_en_inp, train_de_inp, val_de_inp, train_de_out, val_de_out = train_test_split( 
    encoder_input, decoder_input, decoder_output, test_size=0.2 )
print(train_en_inp.shape, val_en_inp.shape, train_de_inp.shape, val_de_inp.shape, train_de_out.shape, val_de_out.shape)


#training
import time
BATCH_SIZE=256
TRAIN_BATCHES, VAL_BATCHES = len(train_en_inp)//BATCH_SIZE, len(val_en_inp)//BATCH_SIZE

all_metrics = []
for epoch in range(1,101):

  tloss, tacc, ti, t0 = 0,0,0,time.time()
  for x,y in generator(train_en_inp, train_de_inp, train_de_out, bs = BATCH_SIZE):
    metrics_train = model.train_on_batch(x, y)
    tloss, tacc, ti = tloss+metrics_train[0], tacc+metrics_train[1], ti+1
    print(f'\rTraining Progress: {ti}/{TRAIN_BATCHES} - loss: {tloss/ti:.4f} - acc: {tacc/ti:.4f}', end="")

  vloss, vacc, vi = 0, 0, 0
  for x,y in generator( val_en_inp, val_de_inp, val_de_out, bs = BATCH_SIZE ):
    metrics_val = model.test_on_batch(x, y)
    vloss, vacc, vi = vloss+metrics_val[0], vacc+metrics_val[1], vi+1
    print(f'\rValidation Progress: {vi}/{VAL_BATCHES} - val_loss: {vloss/vi:.4f} - val_acc: {vacc/vi:.4f}', end="")

  all_metrics.append( ( tloss/ti, tacc/ti, vloss/vi, vacc/vi  ) )
  print(f"\rEpoch {epoch:2.0f} - {(time.time()-t0):.0f}s - loss: {tloss/ti:.4f} - acc: {tacc/ti:.4f} - val_loss: {vloss/vi:.4f} - val_acc: {vacc/vi:.4f}")

#printing some curves
epochs = len(all_metrics)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15,5))
ax[0].plot( range(epochs), list(map(lambda x: x[0], all_metrics)), label="Train Loss" )
ax[0].plot( range(epochs), list(map(lambda x: x[2], all_metrics)), label="Val Loss" )
ax[0].legend();ax[0].set_xlabel('Epochs');ax[0].set_ylabel('Loss');ax[0].grid()
ax[0].set_title("Model training and Validation Loss curves")

ax[1].plot( range(epochs), list(map(lambda x: x[1], all_metrics)), label="Train AUC" )
ax[1].plot( range(epochs), list(map(lambda x: x[3], all_metrics)), label="Val AUC" )
ax[1].legend();ax[1].set_xlabel('Epochs');ax[1].set_ylabel('Accuracy');ax[1].grid()
ax[1].set_title("Model Training and Validation Accuracy Curves")
plt.show()


#Preparing model to translate input sentence
#will take encoder inp sequence, token(<sos>), initial_h, initial_c for first time step
#get word_embedding of input token using decoder embeddings
#For each time step t
#	  get context using one-step-attention, (I am also taking alpha and saving in list for visualization)
#	  append context with previous generated word embedding, 
#	  pass it through dense layer
#	  then to a decoder lstm layer with prev hidden and cell state. to get output, new hidden state, new cell state
#	  pass ouput through dense layer with softmax activation
#	  save this in outputs list
#	  now get the argmax of that output which will be now new_token (increase dims to match input of embedding layer)
#	  get word embedding of that token
token = Input( shape=(1,) ) #( _,1 ) #init_token - <sos>
word_embedding = decoder_embedding(token) #( _,1, 50)
s, c = initial_decoder_h, initial_decoder_c

token_new = token
next_token = Lambda( lambda x: K.expand_dims(K.argmax(x,axis=-1), axis=-1), name="next_token")

outputs = []
alphas = []
for _ in range(Ty):
  context  = one_step_attention( hidden_states, s ) #(_,1, 2*M1)

  alpha = getalpha( hidden_states, s ) #(_, Tx, 1)

  context_word = concat_context_word_prev([ context, word_embedding ])
  context_word = dense_context( context_word ) #(_,1, 100)

  output, s, c = decoder_lstm( context_word, initial_state=[s,c] ) #(_,1,M2), (_,M2), (_,M2)
  output = dense_decoder(output) #(_,1, DECODER_VOCAB_SIZE)

  token_new = next_token( output ) # (_, 1)
  word_embedding = decoder_embedding(token_new) # ( _, 1, 50 )

  outputs.append(output) # after loop it will be a list of length Ty (_ , DEOCDER_VOCAB_SIZE) 
  alphas.append(alpha) #after loop it will be a list of length Ty (_, Tx, 1 )

def stack_alphas( x ):
  x = K.stack(x)
  return K.permute_dimensions( x, pattern=(1,2,0,3) )

alphas = Lambda( stack_alphas )(alphas)
outputs = stacker(outputs)
translator = Model( [ encoder_inp, token, initial_decoder_h, initial_decoder_c ], [outputs, alphas] )
print(translator.summary())


def giveTranslatewithattn( sentence ):
  inputs = preprocess_input(sentence).split()
  enc_inp = [[ word2idx_encoder.get( word, UNK_IDX ) for word in inputs ]]
  enc_inp = pad_sequences( enc_inp, maxlen=MAX_ENCODER_SIZE, padding='post', value=0  )
  initial_h = np.zeros( (1, LATENT_DIM_DE) )
  initial_c = np.zeros( (1, LATENT_DIM_DE) )
  init_token = np.array( [[1]] )
  outputs, alphas = translator( [ enc_inp, init_token, initial_h, initial_c ] )
  outputs = np.argmax( outputs, axis=-1 )
  outputs = [ idx2word_decoder[ idx ] for idx in outputs[0] ]
  attn = pd.DataFrame( np.reshape(alphas, (Tx,Ty)),
                      index = [ idx2word_encoder[idx] for idx in enc_inp ], 
                      columns = outputs )
  outputs = [ w for w in outputs if w not in ('<pad>','<eos>') ]
  return " ".join(inputs), " ".join(outputs), attn


inputs, output, attn = giveTranslatewithattn( "i understand that you want to have a party !" )
sns.heatmap( attn, vmin=-1, vmax=1, center=0, cmap=sns.color_palette("Blues"), square=True  )
plt.show()