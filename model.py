# -*- coding: utf-8 -*-
"""
Created on Tue May 17 20:28:23 2022

@author: dript
"""

import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from tqdm import tqdm
import nltk

nltk.download('stopwords')
STOPWORDS = set(stopwords.words('english'))

from nltk.stem import WordNetLemmatizer
import re
stemmer = WordNetLemmatizer()
nltk.download('wordnet')
import os


def text_processor(text_input):
    text_output = []
    
    for text in text_input:
        # Remove all the special characters
        document = re.sub(r'\W', ' ', str(text))
        
        # remove all single characters
        document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
        
        # Remove single characters from the start
        document = re.sub(r'\^[a-zA-Z]\s+', ' ', document) 
        
        # Substituting multiple spaces with single space
        document = re.sub(r'\s+', ' ', document, flags=re.I)
        
        # Removing prefixed 'b'
        document = re.sub(r'^b\s+', '', document)
        
        # Converting to Lowercase
        document = document.lower()
        
        text_output.append(document)
        
    return text_output

@tf.keras.utils.register_keras_serializable()
def text_processor_tf(text_input):
    
    # Remove all the special characters
    document = tf.strings.regex_replace(text_input,r'\W', ' ')
    
    # Remove punctuations and numbers
    document = tf.strings.regex_replace(document,'[^a-zA-Z]', ' ')

    # Single character removal
    document = tf.strings.regex_replace(document,r"\s+[a-zA-Z]\s+", ' ')
    
    # remove all single characters
    document = tf.strings.regex_replace(document,r'\s+[a-zA-Z]\s+', ' ')
    
    # Remove single characters from the start
    document = tf.strings.regex_replace(document,r'\^[a-zA-Z]\s+', ' ') 
    
    # Substituting multiple spaces with single space
    document = tf.strings.regex_replace(document,r'\s+', ' ')
    
    # Removing prefixed 'b'
    document = tf.strings.regex_replace(document,r'^b\s+', '')
    
    # Converting to Lowercase
    document = tf.strings.lower(document)
        
        
    return document


def load_data(path):
    train,test,dev = "train.csv","test.csv","dev.csv"
    
    return {
            "train": pd.read_csv(os.path.join(path,train)),
            "test": pd.read_csv(os.path.join(path,test)),
            "dev": pd.read_csv(os.path.join(path,dev))
        }

loaded_data = load_data("dataset")


# select data columns
for k,v in tqdm(loaded_data.items()):
    temp_d = v[["sentence_text","label"]]
    one_hot = pd.get_dummies(temp_d['label'])
    temp_d = temp_d.drop(columns = ["label"]).join(one_hot)
    
    loaded_data[k] = temp_d
    

# Process Text to calculate total vocab size
for k,v in tqdm(loaded_data.items()):
    v["clean_sent"] = text_processor(v["sentence_text"])
    loaded_data[k] = v
    
tokenizer = Tokenizer(oov_token='<OOV>')
tokenizer.fit_on_texts(loaded_data["train"].clean_sent)
word_index = tokenizer.word_index

def create_tf_dataset(loaded_data):
        
    training_dataset = (
        tf.data.Dataset.from_tensor_slices(
            (
                loaded_data["train"]['clean_sent'],
                loaded_data["train"][list(loaded_data["train"].columns)[1:-1]]
            )
        )
    )
    
    testing_dataset = (
        tf.data.Dataset.from_tensor_slices(
            (
                loaded_data["test"]['clean_sent'],
                loaded_data["test"][list(loaded_data["test"].columns)[1:-1]]
            )
        )
    )
    
    validation_dataset = (
        tf.data.Dataset.from_tensor_slices(
            (
                loaded_data["dev"]['clean_sent'],
                loaded_data["dev"][list(loaded_data["dev"].columns)[1:-1]]
            )
        )
    )
    
    return training_dataset,testing_dataset,validation_dataset

training_dataset,testing_dataset,validation_dataset = create_tf_dataset(loaded_data)

BUFFER_SIZE = 10000
BATCH_SIZE = 64

training_dataset = training_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
testing_dataset = testing_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
validation_dataset = validation_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


## check the result
for features_tensor, target_tensor in training_dataset.take(1):
    print(f'features:{features_tensor} target:{target_tensor}')

    
## plot sen length
loaded_data['train'].clean_sent.apply(lambda x: len(x.split(" "))).plot(kind = "hist")
#mean_sent_len = loaded_data['train'].clean_sent.apply(lambda x: len(x.split(" "))).mean()
#mean_sent_len = np.floor(mean_sent_len + mean_sent_len * 0.5)

## text vectorization
vectorize_layer = tf.keras.layers.TextVectorization(
    standardize=text_processor_tf,
    max_tokens=len(word_index) + 1,
    output_mode='int',
    output_sequence_length=50)


text_ds = training_dataset.map(lambda x, y: x)
vectorize_layer.adapt(text_ds)



def get_embeddings_matrix(pre_matrix_path,vocab_size):
    embeddings_dictionary = dict()

    glove_file = open(pre_matrix_path, encoding="utf8")
    
    for line in glove_file:
        records = line.split()
        word = records[0]
        vector_dimensions = np.asarray(records[1:], dtype='float32')
        embeddings_dictionary[word] = vector_dimensions
    glove_file.close()
    
    embedding_matrix = np.zeros((vocab_size, 200))
    for word, index in tokenizer.word_index.items():
        embedding_vector = embeddings_dictionary.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector
            
    return embedding_matrix

embedding_matrix = get_embeddings_matrix(os.path.join("glove","glove.6B.200d.txt"),len(word_index) + 1)


model = tf.keras.Sequential([
    vectorize_layer,
    tf.keras.layers.Embedding(input_dim=len(vectorize_layer.get_vocabulary()),
        output_dim=200,
        weights=[embedding_matrix], trainable=False,
        # Use masking to handle the variable sequence lengths
        mask_zero=True),
    tf.keras.layers.LSTM(128),
    tf.keras.layers.Dense(5, activation='sigmoid')
])


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[tf.keras.metrics.AUC(),"acc"])

history = model.fit(training_dataset, epochs=5,
                    validation_data=testing_dataset,
                    validation_steps=1)


model.evaluate(validation_dataset)

r = model.predict(["However, the evolutionary forces that lead to the accumulation of such incompatibilities between diverging taxa are poorly understood."])
print(list(loaded_data["dev"].columns)[1:-1][np.argmax(r)])

model.save('models/classification_model_1')


## checking model is loading successfully or not
new_model = tf.keras.models.load_model('models/classification_model_1')

# Check its architecture
new_model.summary()

new_model.evaluate(validation_dataset)


dot_img_file = 'classification_model_1.png'
tf.keras.utils.plot_model(new_model, to_file=dot_img_file, show_shapes=True)
