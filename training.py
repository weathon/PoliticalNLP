import json
import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import csv

import numpy as np
import matplotlib.pyplot as plt
import os
import random

# with open("ExtractedTweets.csv", 'r', encoding='utf-8') as f:
#     datastore = f.read()

tags=[]
texts=[]
# for i in datastore.split("\n")[1:]:
#     row=i.split(",")
#     # print(row)
#     if row[0]=="":
#         continue
#     tags.append(int(row[0]))
#     text=",".join(row[2:])
#     texts.append(text)

words=[]

for i in os.listdir("./Rep"):
    with open("./Rep/"+i,"r", encoding='utf-8') as f:
        for j in f.read().split("."):
            # j=j.replace("?"," ").replace(","," ").replace("!"," ")
            j=j.replace("?"," ").replace(","," ").replace("!"," ").replace(";"," ").replace("\""," ").replace("\n"," ").replace("\r"," ")

            for k in j.split(" "):
                if not k in words:
                    words.append(k)
            j+="."
            tags.append(1)
            texts.append(j)

for i in os.listdir("./Dem"):
    with open("./Dem/"+i,"r", encoding='utf-8') as f:
        for j in f.read().split("."):
            j=j.replace("?"," ").replace(","," ").replace("!"," ").replace(";"," ").replace("\""," ").replace("\n"," ").replace("\r"," ")

            for k in j.split(" "):
                if not k in words:
                    words.append(k)
            j+="."
            tags.append(1)
            texts.append(j)

#  size budui 

end=len(texts)
for i in range(len(texts)):
    string=""
    for length in range(random.randint(5,15)):
        string+=random.choice(words)+" "
    tags.append(0)
    texts.append(string)
    # print(string)


for i in range(int(len(texts)/0.3)):
    a=random.randint(0,len(texts)-1)
    b=random.randint(0,len(texts)-1)
    tags[a],tags[b]=tags[b],tags[a]
    texts[a],texts[b]=texts[b],texts[a]


print(texts[end+1])
vocab_size = int(len(words)*0.7)
embedding_dim = 2

max_length = 100
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"
training_size = int(len(tags)*0.8)
training_sentences = texts[0:training_size]
testing_sentences = texts[training_size:]
training_labels = tags[0:training_size]
testing_labels = tags[training_size:]

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)

word_index = tokenizer.word_index

training_sequences = tokenizer.texts_to_sequences(training_sentences)
training_padded = pad_sequences(training_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

training_padded = np.array(training_padded)
training_labels = np.array(training_labels)
testing_padded = np.array(testing_padded)
testing_labels = np.array(testing_labels)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(24, activation='relu'),
    
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

model.summary()

num_epochs = 10
history = model.fit(training_padded, training_labels, epochs=num_epochs, validation_data=(testing_padded, testing_labels), verbose=2)


def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.plot(history.history['val_'+string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  plt.show()
  
plot_graphs(history, "accuracy")
plot_graphs(history, "loss")

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_sentence(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

# print(decode_sentence(training_padded[0]))
# print(training_sentences[2])
# print(tags[2])

e = model.layers[0]
weights = e.get_weights()[0]
print(weights.shape) # shape: (vocab_size, embedding_dim)

import io

out_v = io.open('vecs.tsv', 'w', encoding='utf-8')
out_m = io.open('meta.tsv', 'w', encoding='utf-8')
for word_num in range(1, vocab_size):
  word = reverse_word_index[word_num]
  embeddings = weights[word_num]
  out_m.write(word + "\n")
  out_v.write('\t'.join([str(x) for x in embeddings]) + "\n")
out_v.close()
out_m.close()


# sentence = ["We should cut the taxes. The left wing is trying to take away all the money from the people."]

while 1:
    s=input("> ")
    sequences = tokenizer.texts_to_sequences([s])
    padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
    print(model.predict(padded))