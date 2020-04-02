import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Dense
from tensorflow.keras.models import Model

data = pd.read_csv( 'spa.txt', sep='\t', header=None )

NUM_SAMPLES = 10000
data = data.iloc[:NUM_SAMPLES,:2]
data.columns = [ 'input', 'target' ]
print(data.shape)

data['target_input'] = data['target'].apply( lambda x: '<sos> '+x.lower() )
data['target'] = data['target'].apply( lambda x: x.lower()+' <eos>' )
data['input'] = data.input.str.lower()
print(data.head(2))

tokenizer_inputs = Tokenizer()
tokenizer_inputs.fit_on_texts( data.input.values )
input_sequences = tokenizer_inputs.texts_to_sequences( data.input.values )

word2idx_input = tokenizer_inputs.word_index
MAX_INPUT_VOCAB_LEN = len(word2idx_input)+1
print("Max input vocab length ",MAX_INPUT_VOCAB_LEN)

tokenizer_targets = Tokenizer(filters="!.")
tokenizer_targets.fit_on_texts( np.append(data['target_input'], data['target']) )
target_sequences  = tokenizer_targets.texts_to_sequences( data.target.values )
target_input_sequences = tokenizer_targets.texts_to_sequences( data.target_input.values )

word2idx_target = tokenizer_targets.word_index
idx2word_target = { idx:word for word, idx in word2idx_target.items() }

MAX_TARGET_VOCAB_LEN = len(word2idx_target)+1
print("Max target vocab length ",MAX_TARGET_VOCAB_LEN)

MAX_INPUT_SEQ_LEN = max( len(x) for x in input_sequences )
MAX_TARGET_SEQ_LEN = max( len(x) for x in target_sequences )
print("max input seq len ", MAX_INPUT_SEQ_LEN, "max target seq len", MAX_TARGET_SEQ_LEN)

encoder_inputs = pad_sequences( input_sequences, maxlen=MAX_INPUT_SEQ_LEN)
decoder_output = pad_sequences( target_sequences, maxlen=MAX_TARGET_SEQ_LEN, padding='post' )
decoder_input = pad_sequences( target_input_sequences, maxlen=MAX_TARGET_SEQ_LEN, padding='post' )

decoder_output_ohe = np.zeros( (len(decoder_output), MAX_TARGET_SEQ_LEN, MAX_TARGET_VOCAB_LEN) )
for i in range(len(decoder_output)):
  for j in range(MAX_TARGET_SEQ_LEN):
    decoder_output_ohe[ i, j, decoder_output[i,j] ] = 1
decoder_output_ohe.shape

EMBEDDING_DIM = 50
print('Loading word vectors...')
word2vec = {}
with open(f"glove.6B.{EMBEDDING_DIM}d.txt") as fp:
  for line in fp.readlines():
    values = line.split()
    word = values[0]
    vec = np.array(list(map(float,values[1:])))
    word2vec[word] = vec
print("total word vectors ", len(word2vec))

print("Preparing Embedding Matrix")
embedding_matrix = np.zeros( ( MAX_INPUT_VOCAB_LEN, EMBEDDING_DIM ) )
for word,idx in word2idx_input.items():
  embedding = word2idx_input.get(word)
  if embedding is not None:
    embedding_matrix[idx] = embedding

LATENT_DIM = 50

inp_encoder = Input( shape=(MAX_INPUT_SEQ_LEN,) ) #(BATCH_SIZE, MAX_INPUT_SEQ_LEN)
inp_decoder = Input( shape=(MAX_TARGET_SEQ_LEN, ) ) #(BATCH_SIZE, MAX_TARGET_SEQ_LEN)

encoder_embedding = Embedding( MAX_INPUT_VOCAB_LEN, EMBEDDING_DIM, weights=[embedding_matrix], trainable=False )
input_emb = encoder_embedding( inp_encoder ) #(BATCH_SIZE, MAX_INPUT_SEQ_LEN, EMBEDDING_DIM)

encoderlstm = LSTM( LATENT_DIM, return_state=True )
_, h0, c0 = encoderlstm( input_emb ) # _, (BATCH_SIZE, LATENT_DIM), (BATCH_SIZE, LATENT_DIM) 

decoder_embedding = Embedding( MAX_TARGET_VOCAB_LEN, EMBEDDING_DIM )
target_emb = decoder_embedding( inp_decoder ) #(BATCH_SIZE, MAX_TARGET_SEQ_LEN, EMBEDDING_DIM)

decoderlstm = LSTM( LATENT_DIM, return_sequences=True, return_state=True )
targets, _, _ = decoderlstm( target_emb, initial_state=[ h0, c0 ]  ) # (BATCH_SIZE, MAX_TARGET_SEQ_LEN, LATENT_DIM)

dense = Dense( MAX_TARGET_VOCAB_LEN, activation='softmax' )
output = dense( targets ) #(BATCH_SIZE, MAX_TARGET_SEQ_LEN, MAX_TARGET_VOCAB_LEN)

model = Model( [inp_encoder, inp_decoder], output )
model.compile( optimizer="rmsprop", loss="categorical_crossentropy", metrics=['accuracy'] )
print(model.summary())
history = model.fit( [ encoder_inputs, decoder_input ],  decoder_output_ohe, 
                    batch_size=64, epochs=100, validation_split=0.2 )


fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15,5))
ax[0].plot( range(100), history.history['loss'], label="Train Loss" )
ax[0].plot( range(100), history.history['val_loss'], label="Val Loss" )
ax[0].legend();ax[0].set_xlabel('Epochs');ax[0].set_ylabel('Loss');ax[0].grid()
ax[0].set_title("Model training and Validation Loss curves")

ax[1].plot( range(100), history.history['accuracy'], label="Train ACC" )
ax[1].plot( range(100), history.history['val_accuracy'], label="Val ACC" )
ax[1].legend();ax[1].set_xlabel('Epochs');ax[1].set_ylabel('Accuracy');ax[1].grid()
ax[1].set_title("Model Training and Validation Accuracy Curves")
plt.show()

thoughtvector = Model( inp_encoder, [ h0, c0 ] )
inp_word = Input( shape=(1,) ) # (1, 1)
initial_h = Input( shape=(LATENT_DIM,) ) #(1, LATENT_DIM)
initial_c = Input( shape=(LATENT_DIM,) ) #(1, LATENT_DIM)
emb = decoder_embedding(inp_word) #(1, 1, EMBEDDING_DIM)
target, h, c = decoderlstm(emb, initial_state=[ initial_h, initial_c ]) #(1, 1, LATENT_DIM), (1, LATENT_DIM), (1, lATENT_DIM)
out = dense(target)
generator = Model( [ inp_word, initial_h, initial_c ], [ out, h, c ] )
print(generator.summary())

def translate(sentence):
  test_inp = tokenizer_inputs.texts_to_sequences([sentence])
  test_inp = pad_sequences( test_inp, maxlen = MAX_INPUT_SEQ_LEN )
  h,c = thoughtvector.predict(test_inp)

  eos = word2idx_target['<eos>']
  initial_input = np.array([ [word2idx_target['<sos>']] ])
  outputs = []
  for _ in range(MAX_TARGET_SEQ_LEN):
    probs, h, c = generator( [ initial_input, h, c ] )
    idx = np.argmax(probs)
    if idx==0:
      print('wtf')
      probs[0] = 0
      idx = np.argmax(probs)
    if idx==eos: break
    outputs.append(idx2word_target[idx])
    initial_input[0,0] = idx
  return sentence+' -> '+" ".join(outputs)


while True:
  sentence = data.input[np.random.choice(len(data.input))]
  print( translate(sentence.lower()) )
  print('-----------------')
  if input('continue?(y/n)')=='n':break