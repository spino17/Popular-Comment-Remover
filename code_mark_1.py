import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
import json

"""

whether the comment was removed '1' or not '0'

"""

# hyperparameters initialization : could be tweaked
np.random.seed(7)
max_sequence_length = 200
batch_size = 64

# load the dataset
data = pd.read_csv('reddit_train.csv', usecols = ['BODY','REMOVED'], dtype={'BODY': str, 'REMOVE': int})
sequence_data = data.iloc[:20179, 0].tolist()
labels = data.iloc[:20179, -1].tolist()
labels = np.array(labels).astype(np.int64)

# convert text sequences into integer sequences
t = Tokenizer(lower = True)
t.fit_on_texts(sequence_data)
word_to_id = t.word_index # vocabulary
with open('word_integer_map.json', 'w') as f:
    json.dump(word_to_id, f)
integer_encoded_data = t.texts_to_sequences(sequence_data)
padded_data = sequence.pad_sequences(integer_encoded_data, maxlen=max_sequence_length)
x_train, x_test, y_train, y_test = train_test_split(padded_data, labels, test_size = 0.25, random_state = 1)


# built the model
embedding_dim = 32 # embed the word in 32 dimensional vectorized representation
num_vocab = len(t.word_index) + 1 # number of words in vocabulary
model = Sequential()
model.add(Embedding(num_vocab, embedding_dim, input_length=max_sequence_length))
model.add(Dropout(0.2))
model.add(LSTM(100))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=3, batch_size=batch_size)
model.save_weights('trained_model.h5')
model.save('my_model.h5')

# performance check
scores = model.evaluate(x_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

# prediction for the input sequences
model.load_weights('trained_model.h5')
model.compile(loss='mse', optimizer='adam')


test_sequence = "this is reddit" # comes from the front end
temp = []
for word in test_sequence.split(" "):
   temp.append(word_to_id[word])
temp_padded = sequence.pad_sequences([temp], maxlen=max_sequence_length)
pred_score = model.predict(np.array([temp_padded][0]))[0][0]
print("the score for the comment is %s" % (pred_score))

