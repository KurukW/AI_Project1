import matplotlib.pyplot as plt
import cv2
import numpy as np
import seaborn as sns #On ne sait jamais que ça serve
import pandas as pd
from sklearn.preprocessing import StandardScaler, normalize , MinMaxScaler #Normalisation
labels_csv = pd.read_csv('DATA\\labels.csv')
for line in labels_csv.values:
    label = line[0]
    name = line[1]
    #print(f"Le label est '{label}' et le nom de la vidéo : '{name}'")
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
X_link = []
y = []
for label,name in labels_csv.values:
    X_link.append(name)
    y.append(label)
def get_imgs_from_path(path,fps = -1):
    '''
    Retourne une liste d'images. La liste d'image a le nombre de fps voulu
    '''
    cap = cv2.VideoCapture(path)
    fps_actu = cap.get(cv2.CAP_PROP_FPS)
    if fps <= -1: fps = fps_actu #Je peux ne pas donner de fps et ça va prendre le nombre d'fps initial
    ecart_voulu = int(1000/fps)
    ecart_initial = int(1000/fps_actu)
    imgs = []


    while(cap.isOpened()):
        ret, frame = cap.read()

        if ret: #Sinon ça plante quand il n'y a plus d'images

            #Bonne couleur
            frame_RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


            #Récupère seulement certaines images
            t_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            modulo = t_ms % ecart_voulu
            if modulo < ecart_initial:
                #Isoler les images
                imgs.append(frame_RGB)

        else: #Va jusqu'au bout de la vidéo
            break
    else:
        print("Le fichier n'a pas pu être ouvert")
    cap.release()

    return imgs

def get_mov_imgs_from_path(path,fps = -1,color = 'gray'):
    '''
    Retourne une liste d'images. La liste d'image a le nombre de fps voulu
    reprend que le mouvement
    '''
    cap = cv2.VideoCapture(path)
    fps_actu = cap.get(cv2.CAP_PROP_FPS)
    print(f"Traitement de la video n° {path}       ",end="\r")
    if fps <= -1: fps = fps_actu #Je peux ne pas donner de fps et ça va prendre le nombre d'fps initial
    ecart_voulu = int(1000/fps)
    ecart_initial = int(1000/fps_actu)
    imgs = []

    ret, frame = cap.read()

    while(cap.isOpened()):
        prev = frame
        ret, frame = cap.read()

        if ret: #Sinon ça plante quand il n'y a plus d'images
            #Récupère seulement certaines images
            t_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            modulo = t_ms % ecart_voulu
            if modulo < ecart_initial:
                #Isoler les images

                diff = cv2.absdiff(frame,prev)
                if color == 'gray':
                    diff_gray = cv2.cvtColor(diff,cv2.COLOR_BGR2GRAY)
                elif color == 'rgb':
                    diff_gray = diff
                #Normalisation foireuse pour avoir une bonne image

                max_ = np.max(diff_gray)
                if max_ == 0:
                    max_ = 1
                ratio = 255.0 / max_
                diff_gray = diff_gray * ratio


                #_,diff_thresh = cv2.threshold(diff_gray,15,255,cv2.THRESH_BINARY)
                #diff_thresh, c'est juste le mouvement Ici on isole l'image en couleur
                #frame_rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                #J'isole le mouvement de l'image initiale pour avoir une sortie en couleur
                #mov = cv2.bitwise_and(frame_rgb,frame_rgb,mask = diff_thresh)

                #On va travailler avec des images en nuance de gris, c'est bcp plus simple
                imgs.append(diff_gray)

        else: #Va jusqu'au bout de la vidéo
            break
    else:
        print("Le fichier n'a pas pu être ouvert")
    cap.release()

    return imgs
mov_imgs = get_mov_imgs_from_path('DATA\\Videos\\video_91.avi',-1)
#mov_imgs = resize_imgs(mov_imgs,(160,120))

plt.figure(figsize=(20,60)) #(20,60) pour des images en (120,160). (20,180) pour des images en (480,640)
print(mov_imgs[1].shape)
columns = 7
for i, img in enumerate(mov_imgs):
    plt.subplot(int(len(img) / columns + 1), columns, i + 1)
    plt.imshow(img)

labels_file = pd.read_csv('DATA\\labels_list.csv')
labels = labels_file.values

def resize_imgs(imgs,nsize):
    '''
    Change la taille de l'image.
    Le premier élément de nsize est la longueur (width), le deuxième la hauteur (height)
    '''
   # new_imgs = []
  #  for img in imgs:
 #       new_imgs.append(cv2.resize(img, dsize=nsize, interpolation=cv2.INTER_CUBIC))
#    return new_imgs
    return [cv2.resize(img, dsize=nsize, interpolation=cv2.INTER_CUBIC) for img in imgs]

def reduce_fps(imgs,ratio = 6):
    '''
    imgs est une liste d'images
    1/ratio est le nombre d'images qu'on garde
    '''
    new_imgs = []
    for i,img in enumerate(imgs):
        if i % ratio == 0:
            new_imgs.append(img)
    return new_imgs

def scale(X, x_min, x_max):
#     nom = (X-X.min(axis=0))*(x_max-x_min)
#     denom = X.max(axis=0) - X.min(axis=0)
#     denom[denom==0] = 1

    nom = (X-np.min(X))*(x_max-x_min)
    denom = np.max(X) - np.min(X)
    if denom == 0:
        denom = 1
    return x_min + nom/denom
def scale_by_pixels(img, x_min, x_max):
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

def normalize_imgs(imgs):
    norm_imgs = []
    for img in imgs:
        norm_img = scale_by_pixels(img,0,1)
        #norm_img = [scale(line,0,1) for line in img]
#         scaler = MinMaxScaler(copy = False) #Askip ça prend moins de mémoires, jsp si ça a des inconvénients
#         scaler.fit(img)
#         norm_img = scaler.transform(img)
        norm_imgs.append(norm_img)
#         print(scaler.get_params())
#     print(f"Après normalisation, max = {np.max(norm_imgs)} et min = {np.min(norm_imgs)}")
    return norm_imgs

#Récupère toutes les vidéos, prend slmt z fps et les resize
import time
folder_path = 'Data\\Videos\\'
fps = 10
size = (160,120) # La shape initial est (480, 640, 3), Il faut inverser le sens de with et height.
#Size demi par rapport à initial
nb_images_max = 10 #Prend les x premières images

start = time.time()
X = []
for i,name in enumerate(X_link):
    if i >= nb_images_max:
        break

    file_path = folder_path + name
    imgs_of_video = get_mov_imgs_from_path(file_path,fps,'gray')
    resized_imgs = resize_imgs(imgs_of_video, size)
    X.append(resized_imgs)

end = time.time()
y = y[:len(X)]

print(f"ça a prit {end-start} secondes",end=" "*15)

#Melanger les videos associés aux labels
import random
both = list(zip(X,y))
random.shuffle(both)
X,y = zip(*both)
print(y)
plt.imshow(X[-1][10])
nb_classes = len(set(y))
print(nb_classes)

#Labels
labels_csv = pd.read_csv("DATA\\labels.csv")
labels_name = pd.read_csv("DATA\\labels_list.csv")
labels_tests_csv = pd.read_csv("DATA\\labels_tests.csv")

#Melange la liste
labels_val = list(labels_csv.values)
random.shuffle(labels_val) #Il faut absolument shuffle pour train
labels_tests = list(labels_tests_csv.values)

#Créé un dictionnaire qui associe le nom du label et un numérique afin de le mettre dans le modele
labels_n = {}
for i,label in enumerate(labels_name.values):
    labels_n[label[0]] = i
    labels_n[i] = label[0]

def my_gen(labels_list, folder_path, batch_size = 10, pointer = 0, fps = 5, size = (160,120)):
    '''
    à partir de la liste des labels et de la position du foler
    Retourne:
    -une liste d'images de vidéos de shape (batch_size,n_frames,height,width[,channels])
        n_frames = fps*2.4
    -Une liste de numérique avec les labels correspondant aux images (de len = batch_size)
    -Un pointer qui permet de faire tourner my_gen à nouveau et recevoir les éléments suivants de la liste


    Normalement on peut envoyer X et y_num dans le modele directement (avec fit ou train_on_batch, les deux devraient fonctionner)
    Je vais essayer de travailler avec des thread pour avoir un thread en Train et un trhead en gen
    '''

    y_num = []
    X = []
    for i in range(pointer, batch_size + pointer):
        pointer = i+1
        if i >= len(labels_list):
            break
        label, video_name = labels_list[i]
        #Label en numérique
        y_num.append(labels_n[label])

        #Preprocess d'images
        file_path = folder_path + "\\" + video_name
        imgs_of_video = get_mov_imgs_from_path(file_path,fps,'gray') #Gray ou rgb
        resized_imgs = resize_imgs(imgs_of_video, size)
        norm_imgs = normalize_imgs(resized_imgs)





        X.append(norm_imgs)
        #X.append(np.array(resized_imgs)) #Je ne suis pas obligé de transformer en array tt de suite, ça va se faire par la suite
    X = np.array(X)
    y_num = np.array(y_num)




#     xmax=X.max()
#     if xmax == 0:
#         xmax = 1
#     X /= xmax
#     xmin = X.min()
#     if xmin <0:
#         X -= xmin
#         xmax=X.max()
#         if xmax == 0:
#             xmax = 1
#         X /= xmax
    return (X, y_num, pointer)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential
from tensorflow.keras.layers import  Dense, Conv3D, BatchNormalization,MaxPooling3D, Dropout, LSTM

sample_shape = (24,120,160,1) #width = 160, height = 120, nframes = 24, 3 channels si on est en RGB (si on est en gris on sait pas)
model = Sequential()
model.add(Conv3D(128, strides =(1,1,1), padding="same", kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=sample_shape,data_format='channels_first'))
#model.add(Conv3D(32, strides =(1,1,1), padding="same", kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=sample_shape,data_format='channels_last'))
model.add(Dense(1, activation='relu', kernel_initializer='he_uniform'))
model.add(BatchNormalization(center=True, scale=True))
model.add(Conv3D(64, strides=(1,1,1), padding="same", kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(1, activation='relu', kernel_initializer='he_uniform'))
model.add(BatchNormalization(center=True, scale=True))
model.add(MaxPooling3D(pool_size=(3, 3, 3)))
model.add(Dropout(0.2))

model.add(Conv3D(128, strides=(1,1,1), padding="same", kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(1, activation='relu', kernel_initializer='he_uniform'))
model.add(BatchNormalization(center=True, scale=True))
model.add(Conv3D(128, strides=(1,1,1), padding="same", kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(1, activation='relu', kernel_initializer='he_uniform'))
model.add(BatchNormalization(center=True, scale=True))
model.add(MaxPooling3D(pool_size=(3, 3, 3)))
model.add(Dropout(0.2))

model.add(Conv3D(64, strides=(1,1,1), padding="same", kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(1, activation='relu', kernel_initializer='he_uniform'))
model.add(BatchNormalization(center=True, scale=True))
model.add(Conv3D(64, strides=(1,1,1), padding="same", kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(1, activation='relu', kernel_initializer='he_uniform'))
model.add(BatchNormalization(center=True, scale=True))
model.add(MaxPooling3D(pool_size=(3, 3, 3)))
model.add(Dropout(0.2))

model.add(Conv3D(32, strides=(1,1,1), padding="same", kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(32, activation='relu', kernel_initializer='he_uniform'))
model.add(BatchNormalization(center=True, scale=True))

#model.add(Dense(1, activation='relu', kernel_initializer='he_uniform'))
#model.add(MaxPooling3D(pool_size=(1, 3, 3)))
#model.add(layers.Reshape((128, 5120),input_shape = (1,128,4,5,256)))
#model.add(layers.Reshape((128, 640),input_shape = (1,128,4,5,32)))
#model.add(layers.Reshape((1,640),input_shape = (1,4,5,32)))
model.add(layers.Reshape((1,2560)))
#model.add(layers.Flatten())

#LSTM
#model.add(Embedding(num_distinct_words, embedding_output_dims, input_length=max_sequence_length))
model.add(LSTM(units = 2,activation="tanh", recurrent_activation="sigmoid",return_sequences = False))
#[None, 128, 4, 5, 256]


#softmax
model.add(Dense(13, activation='softmax'))

model.summary()

X_np = np.array(X)
y_np = np.array(y)
X_np = X_np.reshape(nb_images_max,24,120,160,1)
model.compile(optimizer='adam',
              loss = tf.keras.losses.categorical_crossentropy,
              metrics=['accuracy'])


#Il vaut mieux recompiler le modele avant d'exécuter ce code

from tensorflow.keras.utils import to_categorical
#Parametres
batch_size = 2
fps = 10 #Pour une shape de 12 (à définir avant de faire le modele)
size = (160,120)
pointer = 0
can_continue = True
nb_classes = 13

while can_continue:
    X1_not_expand, y_num1, pointer = my_gen(labels_val, "DATA\\Videos",batch_size = batch_size,fps = fps, size = size, pointer = pointer)

    if pointer >= len(labels_val):
        can_continue = False
    print()
    print('le pointeur vaut ',pointer)  #DEBUG

    X1 = np.expand_dims(X1_not_expand, axis=len(X1_not_expand.shape)) #Ajoute une dimension
    print(y_num1)
    y_num1_categorical = to_categorical(y_num1, num_classes = nb_classes)

    print('vidéos chargées, shape:',X1.shape,'. y len : ',len(y_num1_categorical)) #DEBUG
    print('fit is busy') #DEBUG
    #print('type de x: ',type(X1),' . Type de y: ',type(y_num1_categorical))
    with tf.device('/gpu:0'):
        model.fit(X1, y_num1_categorical,
                batch_size = 2,
                epochs=2,
                verbose=1,
                validation_split = 0)

    #osef du warning
