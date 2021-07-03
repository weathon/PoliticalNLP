import json
# import tensorflow as tf

# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences
import csv

vocab_size = 10000
embedding_dim = 16
max_length = 100
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"
training_size = 20000
with open("ExtractedTweets.csv", 'r') as f:
    datastore = f.read()

tags=[]
texts=[]
for i in f.split("\n")[1:]:
    row=i.split(",")
    tags.append(row[0])
    text=",".join(row[2:])
    texts.append(text)

