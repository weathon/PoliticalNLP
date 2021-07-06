import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
import numpy as np
import matplotlib.pyplot as plt

def split_input_target(sequence):
    input_text = sequence[:-1]
    target_text = sequence[1:]
    return input_text, target_text

text=open("pall.txt","r").read()

vocab = sorted(set(text.split(" ")))
print(f'{len(vocab)} unique words')

#switched places, after yas
vocab = sorted(set(text))
print(f'{len(vocab)} unique characters')



ids_from_chars = preprocessing.StringLookup(
    vocabulary=list(vocab), mask_token=None)

all_ids = ids_from_chars(tf.strings.unicode_split(text, 'UTF-8'))
ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)

seq_length = 82
examples_per_epoch = len(text)//(seq_length+1)


sequences = ids_dataset.batch(seq_length+1, drop_remainder=True)
# zhe yang fu zhi dai ma zhen meiyi si 

dataset = sequences.map(split_input_target)

BATCH_SIZE = 64 
BUFFER_SIZE = 10000

dataset = (
    dataset
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE, drop_remainder=True)
    .prefetch(tf.data.experimental.AUTOTUNE))





# Length of the vocabulary in chars
vocab_size = len(vocab)

# The embedding dimension
embedding_dim = 256

# Number of RNN units
rnn_units = 1024

from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional

model = tf.keras.models.Sequential()
vs=len(ids_from_chars.get_vocabulary())
print(vs)
model.add(Embedding(vs, embedding_dim))
model.add(LSTM(150))
model.add(Dense(vs, activation='softmax'))

# loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)


# example_batch_loss = loss(target_example_batch, example_batch_predictions)
# mean_loss = example_batch_loss.numpy().mean()
# print("Prediction shape: ", example_batch_predictions.shape, " # (batch_size, sequence_length, vocab_size)")
# print("Mean loss:        ", mean_loss)

# model.compile(optimizer='adam', loss=loss)
model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])

EPOCHS = 20
# history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])
history = model.fit(dataset, epochs=EPOCHS)

print(model)


def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.show()

plot_graphs(history, 'accuracy')