'''
Dans ce script python on va fabriquer notre modèle.
On utilise les librairies Tensorflow et Keras (comme en cours).
Elles ont des RNN intégrées qui vont nous servir à analyser les vidéos.


ça peut être bien de faire le modèle sur jupyter
afin de pouvoir visualier chaque étape plus facilement
'''

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential
from tensorflow.keras.layers import  Dense, Conv3D, BatchNormalization,MaxPooling3D, Dropout, LSTM
import matplotlib.pyplot as plt
import cv2
import numpy as np
import seaborn as sns #On ne sait jamais que ça serve
import pandas as pd #Même argument

'''
Insertion des données
'''
#images.open
#cv2.imread
#Insertion des vidéos?
#https://www.youtube.com/watch?v=fmga0i0MXuU&ab_channel=Weights%26Biases
#à 3 minutes, il montre un générateur.
#Le principe est d'amener les vidéos par paquet et non tout en un coup
#Sinon ça prend trop de place dans la mémoire
#Il combine des modèles aussi mais je n'ai pas vraiment suivi

'''
Traitement des données
'''
#Est-ce qu'il faut fixer la durée de la vidéo?
#Le réseau neuronale prend tout seul les images qui l'intéressent (nb de FPS)

'''
Fabrication/Design du modèle
'''
#Inspiration des plus grands
#CNN?
#LSTM(RNN)?
sample_shape = (1,120,160,12)
model = Sequential()
model.add(Conv3D(32, strides =(1,1,1), padding="same", kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=sample_shape,data_format='channels_first'))
model.add(Dense(256, activation='relu', kernel_initializer='he_uniform'))
model.add(BatchNormalization(center=True, scale=True))
model.add(Conv3D(32, strides=(1,1,1), padding="same", kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=sample_shape,data_format='channels_first'))
model.add(Dense(256, activation='relu', kernel_initializer='he_uniform'))
model.add(BatchNormalization(center=True, scale=True))
model.add(MaxPooling3D(pool_size=(3, 3, 3)))
model.add(Dropout(0.2))

model.add(Conv3D(64, strides=(1,1,1), padding="same", kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=sample_shape,data_format='channels_first'))
model.add(Dense(256, activation='relu', kernel_initializer='he_uniform'))
model.add(BatchNormalization(center=True, scale=True))
model.add(Conv3D(64, strides=(1,1,1), padding="same", kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=sample_shape,data_format='channels_first'))
model.add(Dense(256, activation='relu', kernel_initializer='he_uniform'))
model.add(BatchNormalization(center=True, scale=True))
model.add(MaxPooling3D(pool_size=(3, 3, 3)))
model.add(Dropout(0.2))

model.add(Conv3D(64, strides=(1,1,1), padding="same", kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=sample_shape,data_format='channels_first'))
model.add(Dense(256, activation='relu', kernel_initializer='he_uniform'))
model.add(BatchNormalization(center=True, scale=True))
model.add(Conv3D(64, strides=(1,1,1), padding="same", kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=sample_shape,data_format='channels_first'))
model.add(Dense(256, activation='relu', kernel_initializer='he_uniform'))
model.add(BatchNormalization(center=True, scale=True))
model.add(MaxPooling3D(pool_size=(3, 3, 3)))
model.add(Dropout(0.2))

model.add(Conv3D(128, strides=(1,1,1), padding="same", kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=sample_shape,data_format='channels_first'))
model.add(Dense(256, activation='relu', kernel_initializer='he_uniform'))
model.add(BatchNormalization(center=True, scale=True))


#LSTM
#model = Sequential() #Mis au début
#model.add(Embedding(num_distinct_words, embedding_output_dims, input_length=max_sequence_length))
model.add(LSTM(units = 128,activation="tanh", recurrent_activation="sigmoid",return_sequences = False))



#softmax
model.add(Dense(10, activation='softmax'))

# Compile the model
#model.compile(loss='categorical_crossentropy',optimizer=keras.optimizers.Adam(lr=0.001),metrics=['accuracy'])
#model.summary()

'''
Train du modèle
'''
#fit()
# Fit data to model

# history = model.fit(X_train, targets_train,
#             batch_size=128,
#             epochs=40,
#             verbose=1,
#             validation_split=0.3)


'''
Confusion matrix et tests
'''


'''
Enregistrement du modèle dans un fichier pour réutilisation
'''
