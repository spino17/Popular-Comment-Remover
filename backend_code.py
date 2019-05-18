import numpy as np
from keras.models import load_model
from keras.preprocessing import sequence
import json
import keras

max_sequence_length = 200 # maximum limit of comment
model = load_model('my_model.h5') # loads trained compiled model
with open('word_integer_map.json') as f:
    word_to_id = json.load(f) # loads the vocabulary dictionary

# saliency feature map
inp = model.layers[0].input
outp = model.layers[-1].output
saliency = keras.backend.gradients(keras.backend.sum(max_outp), inp)

test_sequence = "i ll everyone" # comment : comes from the front end
temp = []
for word in test_sequence.split(" "):
   temp.append(word_to_id[word])
temp_padded = sequence.pad_sequences([temp], maxlen=max_sequence_length)
pred_score = model.predict(np.array([temp_padded][0]))[0][0]
print("the score for the comment is %s" % (pred_score * 100)) # gives score percentage

def predict_score(test_sequence):
   max_sequence_length = 200
   model = load_model('./my_model.h5')
   with open('./word_integer_map.json') as f:
      word_to_id = json.load(f)

   temp = []
   for word in test_sequence.split(" "):
      temp.append(word_to_id[word])
   temp_padded = sequence.pad_sequences([temp], maxlen=max_sequence_length)
   pred_score = model.predict(np.array([temp_padded][0]))[0][0]
   return pred_score * 100