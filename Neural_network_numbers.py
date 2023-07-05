#!/usr/bin/env python
# coding: utf-8

# In[16]:


import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import linear, relu, sigmoid
from extra_keras_datasets import emnist
from matplotlib import pyplot as plt


# In[4]:


(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()


# In[5]:


len(X_train)


# In[6]:


X_train.shape


# In[7]:


X_train_flattened = X_train.reshape(len(X_train), 28*28)
X_test_flattened = X_test.reshape(len(X_test), 28*28)


# In[12]:


from tensorflow.keras import layers

model = Sequential(
    [               
        tf.keras.Input(shape=(784,)),
        Dense(25, activation='relu'),
        Dense(15, activation='relu'),
        Dense(10, activation='linear')
    ], name = "my_model" 
)


# In[14]:


model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer = 'adam', metrics = ['accuracy']
)

history = model.fit(
    X_train_flattened,y_train,
    epochs=40
)


# In[20]:


sample_letter = X_test_flattened[1000]
image_of_two = X_test[1000]

prediction = model.predict(image_of_two.reshape(1,784))  # prediction

print(f" predicting a Two: \n{prediction}")
print(f" Largest Prediction index: {np.argmax(prediction)}")

sample_letter = X_test[1000]
sample_letter_resize = sample_letter.reshape(28, 28)
plt.imshow(sample_letter_resize, cmap = 'binary')
plt.axis("off")
plt.show()


# In[ ]:




