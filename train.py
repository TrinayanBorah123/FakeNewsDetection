# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import tensorflow as tf
import pandas as pd
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import nltk
import re
from nltk.corpus import stopwords
nltk.download('stopwords')


true_df = pd.read_csv('True.csv')
fake_df = pd.read_csv('Fake.csv')

#Creating 'check' on both dfs that will be the target feature.

true_df['label'] = 1
fake_df['label'] = 0

#We will combine both dfs.

df_news = pd.concat([true_df, fake_df])

#Shuffling to see some Fakes

df_news=df_news.sample(frac = 1)

#Reseting Index
df_news.reset_index(inplace=True)

## Get the Independent Features

X=df_news.drop('label',axis=1)
## Get the Dependent features
y=df_news['label']

### Vocabulary size
voc_size=5000
messages=X.copy()

### Dataset Preprocessing
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
corpus = []
for i in range(0, len(messages)):
    print(i)
    review = re.sub('[^a-zA-Z]', ' ', messages['title'][i])
    review = review.lower()
    review = review.split()
    
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)
    
#Onehot representation
onehot_repr=[one_hot(words,voc_size)for words in corpus] 
onehot_repr

#Embedding Representation
sent_length=20
embedded_docs=pad_sequences(onehot_repr,padding='pre',maxlen=sent_length)
#print(embedded_docs)
from tensorflow.keras.layers import Dropout
embedding_vector_features=40
model=Sequential()
model.add(Embedding(voc_size,embedding_vector_features,input_length=sent_length))
model.add(LSTM(64,return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(32,return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(12))
model.add(Dropout(0.3))
model.add(Dense(1,activation='sigmoid'))
opt = Adam(learning_rate=0.0001)
model.compile(loss='binary_crossentropy',optimizer=opt,metrics=['accuracy'])
#print(model.summary())

import numpy as np
X_final=np.array(embedded_docs)
y_final=np.array(y)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.33, random_state=42)


### Finally Training
history = model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=20,batch_size=64)

#Save the model
# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")