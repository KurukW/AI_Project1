'''
Dans ce script python on va fabriquer notre modèle.
On utilise les librairies Tensorflow et Keras (comme en cours).
Elles ont des RNN intégrées qui vont nous servir à analyser les vidéos.


ça peut être bien de faire le modèle sur jupyter
afin de pouvoir visualier chaque étape plus facilement
'''

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
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


'''
Train du modèle
'''
#fit()


'''
Confusion matrix et tests
'''


'''
Enregistrement du modèle dans un fichier pour réutilisation
'''
