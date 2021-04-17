'''
Prediction en live de mouvements



A FAIRE:
Ajouter la prédiction et le mettre en thread au besoin


'''

import cv2
import tensorflow as tf
from tensorflow import keras
import numpy as np
import time
import threading
from queue import Queue
import sys #Afin d'arrêter le programme
import pandas as pd


#-------------------------------------------------------------------------------
'''
Fonctions

'''
def scale_by_pixels(img, x_min, x_max):
    '''
    Scale une image entre 0 et 1.
    Méthode pas efficace mais fonctionnelle contrairement à d'autres méthodes
    qui faisaient lignes par lignes ce qui causait des erreurs
    dans le résultat (lignes dans l'image)
    '''
    new_img = img.copy()
    p_min = 1000
    p_max = -1000
    for line in img:
        for pixel in line:
            p_min = min(p_min,pixel)
            p_max = max(p_max,pixel)

    #p_min et p_max sont les min et max totaux de mon image
    for l, line in enumerate(img):
        for p,pixel in enumerate(line):
            nom = (pixel - p_min)*(x_max - x_min)
            denom = (p_max - p_min)
            if denom == 0: denom = 1
            new_img[l][p] = x_min + nom/denom
    return new_img



#Thread
def predict(model):
    '''
    Prédit un résultat de la queue et le remet dans la queue
    '''
    while True:
        X = q_to_pred.get()
        pred = model.predict(X)
        q_pred.put(pred)

def show_prediction():
    '''
    Affiche un résultat
    '''
    while True:
        pred = q_pred.get()
        print("Prediction : ",labels_n[pred.argmax()])
        #print("somme de la prédiction",np.sum(pred)) #DEBUG
        print("Nouvelle prediction:",pred) #DEBUG
        #Si j'ai [nan,nan,...] c'est que le modele est cassé




#-------------------------------------------------------------------------------
'''
Main

'''

#Choix du modele, attention que le modele doit exister
fps = 8
size = (40,30) #Sens inverse au nom du modèle
nb_classes = 10
epochs = 1
batch_size = 20
pack_size = 50
learning_rate = 0.01


#Import du modèle
#Certains modèles sont cassés, il n'y a pas le .pb dedans alors il ne sait pas ouvrir



param = f"_{fps}_{size[1]}_{size[0]}_{nb_classes}_{epochs}_{batch_size}_{pack_size}_{int(learning_rate*1000)}mili"
type = 'model_LSTM'
folder = 'Saved_model'
full_path = folder + '\\' + type + param
try:
    model = keras.models.load_model(full_path)
    #keras.models.load_model('Saved_model\\modele_stolen_compile')
    print("modele importé avec succès")
except:
    print('''Erreur lors du chargement du modele,
    vérifiez que les paramètres sont bons et que le modele existe''')
    print("Je voulais importer le modele:",full_path)
    sys.exit()


cap = cv2.VideoCapture(0)


#Gestion des FPS
fps_actu = cap.get(cv2.CAP_PROP_FPS)
if fps <= -1: fps = fps_actu #Je peux ne pas donner de fps et ça va prendre le nombre d'fps initial
ecart_voulu = int(1000/fps)
ecart_initial = int(1000/fps_actu)
first_start = time.time()


#Fabrication du dictionnaire
labels_name = pd.read_csv("DATA\\labels_uses.csv")
labels_n = {}
for i,label in enumerate(labels_name.values):
    labels_n[label[0]] = i
    labels_n[i] = label[0]


# Automatically grab width and height from video feed
# (returns float which we need to convert to integer for later on!)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#Je le met une fois dehors pour avoir une première image à copier dans prev
ret, frame = cap.read()


#Initialisation des variables
imgs = []
X = []
count = 0
to_mean = 0
n_frames = int(fps*2.4)


#queue
q_to_pred = Queue() #X vers la prédiction
q_pred = Queue() #pred vers l'affichage


#Thread
t_show = threading.Thread(target=show_prediction,daemon=True)
t_show.start()
t_pred = threading.Thread(target=predict,args=((model,)),daemon= True)
t_pred.start()



while True:
    prev = frame
    ret, frame = cap.read()
    start = time.time()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    #Récupère seulement certaines images
    #t_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
    now = time.time()
    t_ms = (now-first_start)*1000
    modulo = t_ms % ecart_voulu
    if modulo < ecart_initial:
        #Isoler les images
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.cvtColor(prev,cv2.COLOR_BGR2GRAY)
        diff_gray = cv2.absdiff(gray,prev_gray)
    #Preprocess avant d'ajouter
        #Réduire la taille
        resized = cv2.resize(diff_gray, dsize=size, interpolation=cv2.INTER_CUBIC)
        #Normaliser
        normalized = scale_by_pixels(resized,0,1)
        imgs.append(normalized)


    cv2.imshow('frame',frame)

    if len(imgs) > n_frames:
        #Récupère les x premières images et supprimes de la liste
        X = imgs[:n_frames]
        del imgs[:n_frames]
        #Traitement de X pour pouvoir prédire
        X = np.array(X)
        X = np.expand_dims(X, axis=len(X.shape)) #Ajoute un channel
        X = np.expand_dims(X, axis = 0) #Ajoute une dimension qui signifie
        #                                qu'on analyse des vidéos et non des images
        #print("Taille de X qu'on va prédire",X.shape)

        start_pred = time.time()
        #Predict X
        q_to_pred.put(X)
        #print(f"Il se passe {start_pred - end_pred} secondes entre deux predictions")

        #pred = model.predict(X)
        end_pred = time.time()






    #Tests des performances
    end = time.time()
    count += 1
    end_start = end-start
    to_mean += end_start
    #print(f"une exécution prend {end_start} secondes")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# When everything done, release the capture and destroy the windows
cap.release()
cv2.destroyAllWindows()
# t_show.join()
# t_pred.join()
#print(f"une exécution prend en moyenne  {to_mean/count} secondes")
