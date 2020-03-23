import numpy as np

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.layers import Input, Embedding, Dense, LSTM
from tensorflow.keras.models import Model


#prepare input/ouput text sequences
input_texts = []
output_texts = []
with open('../datasets/robert_frost.txt',"r") as poem:
  for line in poem.readlines():
    line = line.strip()
    if line == "": continue
    input_texts.append( '<sos> '+line )
    output_texts.append( line + ' <eos>' )
all_lines = input_texts+output_texts
print(len(input_texts))

#tokenize sequences into integer sequences
tokenizer = Tokenizer( filters="." )
tokenizer.fit_on_texts(all_lines)
input_sequences = tokenizer.texts_to_sequences( input_texts )
output_sequences = tokenizer.texts_to_sequences( output_texts )
input_sequences[:5]

word2idx = tokenizer.word_index
MAX_SEQ_LENGTH = max( len(x) for x in input_sequences )
MAX_VOCAB_SIZE = len(word2idx)+1
EMBEDDING_DIM = 100
print( "MAX_SEQ_LENGTH: ", MAX_SEQ_LENGTH, "MAX_VOCAB_SIZE: ",MAX_VOCAB_SIZE )

#make input and output sequences
input_sequences = pad_sequences( input_sequences, maxlen=MAX_SEQ_LENGTH, padding='post' )
output_sequences = pad_sequences( output_sequences, maxlen=MAX_SEQ_LENGTH, padding='post' )

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
embedding_matrix = np.zeros( ( MAX_VOCAB_SIZE, EMBEDDING_DIM ) )
for word,idx in word2idx.items():
  embedding = word2vec.get(word)
  if embedding is not None:
    embedding_matrix[idx] = embedding
embedding_matrix.shape

#one hot encoded targets
ohe_targets = np.zeros( (len(output_sequences), MAX_SEQ_LENGTH, MAX_VOCAB_SIZE)  )
for i in range(len(output_sequences)):
  for j in range(MAX_SEQ_LENGTH):
    ohe_targets[ i, j, output_sequences[i][j] ] = 1
print("Final Targets: ", ohe_targets.shape)



#Learning model through teacher forcing
LSTM_STATE_DIM = 100
input_seq = Input( shape=(MAX_SEQ_LENGTH,) ) # BATCH_SIZE, MAX_SEQ_LENGTH
initial_h = Input( shape=(LSTM_STATE_DIM,) ) #BATCH_SIZE, 100
initial_c = Input( shape=(LSTM_STATE_DIM,) )  #BATCH_SIZE, 100
#generate Embeddings
embedding_layer =  Embedding( MAX_VOCAB_SIZE, EMBEDDING_DIM, weights=[embedding_matrix], trainable=False)
emb = embedding_layer(input_seq) #BATCH_SIZE, MAX_SEQ_LENGTH, EMBEDDING_DIM
#generate output sequence, return_state is required when we will use for testing
lstm = LSTM( LSTM_STATE_DIM, return_sequences=True, return_state=True)
seq, _ , _ = lstm(emb, initial_state=[ initial_h, initial_c ]) #(BATCH_SIZE, MAX_SEQ_LENGTH, 100
#get output
dense = Dense(MAX_VOCAB_SIZE, activation='softmax')
output = dense(seq) #BATCH_SIZE, MAX_SEQ_LENGTH, MAX_VOCAB_SIZE
model = Model( [ input_seq, initial_h, initial_c ] , output)
model.compile( loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'] )
print(model.summary())


print('Training Model')
z = np.zeros( (len(input_sequences), LSTM_STATE_DIM ) )
history = model.fit( [ input_sequences,z,z ], ohe_targets, batch_size=64, epochs=2000, validation_split=0.2, verbose=2 )


#generator Model
input2 = Input( shape=(1,) ) #1, 1 
emb = embedding_layer(input2) #1, 1, EMBEDDING_DIM
x, h, c = lstm(emb, initial_state=[initial_h, initial_c]) # (1, 1, 100)
output2 = dense(x) #1, 1, MAX_VOCAB_SIZE
generator = Model( [input2, initial_h, initial_c], [ output2, h, c ] )
print(generator.summary())

#for index to word mapping
idx2word = { id:w for w,id in word2idx.items() }


#to generate a sample line
def sample_line():
  new_input = np.array( [ [word2idx['<sos>']] ] )
  h = np.zeros( (1, LSTM_STATE_DIM) )
  c = np.zeros( (1, LSTM_STATE_DIM) )

  eos = word2idx['<eos>']
  output_poem = []
  for _ in range( MAX_SEQ_LENGTH ):
    probs,h,c = generator.predict( [new_input, h, c] )
    probs = probs[0,0]
    if np.argmax(probs)==0:
      print('wtf')
    probs[0] = 0
    probs /= probs.sum()
    #choose random next word, based on probability
    idx = np.random.choice( len(probs), p=probs )
    if idx==eos:
      break
    output_poem.append( idx2word[idx]  )
    new_input[0][0] = idx #change new input to new word
  return " ".join(output_poem)


while True:
  for _ in range(4):
    print(sample_line())
  if input("Generate another? (y/n)")=='n':
    break