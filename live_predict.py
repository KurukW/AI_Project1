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


#-------------------------------------------------------------------------------
'''
Main

'''
fps = -1 #Changer le nombre d'FPS ne change pas le temps d'exécution donc
        #mes opérations ne prennent pas de temps
size = (80,60)




cap = cv2.VideoCapture(0)


#Gestion des FPS
fps_actu = cap.get(cv2.CAP_PROP_FPS)
if fps <= -1: fps = fps_actu #Je peux ne pas donner de fps et ça va prendre le nombre d'fps initial
ecart_voulu = int(1000/fps)
ecart_initial = int(1000/fps_actu)




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


while True:
    prev = frame
    ret, frame = cap.read()
    start = time.time()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    #Récupère seulement certaines images
    t_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
    modulo = t_ms % ecart_voulu
    if modulo < ecart_initial:
        #Isoler les images
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.cvtColor(prev,cv2.COLOR_BGR2GRAY)
        diff_gray = cv2.absdiff(gray,prev_gray)
        #Preprocess avant Ajouter
        #Réduire la taille
        resized = cv2.resize(diff_gray, dsize=size, interpolation=cv2.INTER_CUBIC)
        #Normaliser
        normalized = scale_by_pixels(resized,0,1)
        imgs.append(normalized)


    if len(imgs) > 24:
        #Récupère les x premières images et supprimes de la liste
        X = imgs[:24]
        del imgs[:24]

        #Predict X


    cv2.imshow('frame',diff_gray)



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
#print(f"une exécution prend en moyenne  {to_mean/count} secondes")
