import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from numpy.lib.stride_tricks import as_strided
import cv2
from random import randint
from skimage.measure import compare_ssim
from scipy import signal
# ----------------------------------------------------------------------------
'''FONCTIONS '''
#Stolen https://gist.github.com/arifqodari/dfd734cf61b0a4d01fde48f0024d1dc9
def strided_convolution(image, weight, stride):
    im_h, im_w = image.shape
    f_h, f_w = weight.shape

    out_shape = (1 + (im_h - f_h) // stride, 1 + (im_w - f_w) // stride, f_h, f_w)
    out_strides = (image.strides[0] * stride, image.strides[1] * stride, image.strides[0], image.strides[1])
    windows = as_strided(image, shape=out_shape, strides=out_strides)

    return np.tensordot(windows, weight, axes=((2, 3), (0, 1)))


def apply_kernel(img):
    '''
    Applique un kernel à l'image
    '''
    kernel = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]) #Edge detection
    #kernel = np.array([[1,1,1],[1,1,1],[1,1,1]])/9 #Flou
    #kernel = np.ones((5,5),np.uint8)/25 #Grand flou
    #kernel = np.array([[-1,0,-1],[0,4,0],[-1,0,-1]]) #Edge detection 2
    #kernel = np.array([[0,0,0],[0,1,0],[0,0,0]]) #Identity
    #kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]]) #Netteté
    #kernel = np.array([[-1,0,1],[-2,0,2],[-1,0,1]]) #SobelX
    # kernel = np.array([ [0,0,1,0,0],
    #                     [1,1,1,1,1],
    #                     [1,1,1,1,1],
    #                     [1,1,1,1,1],
    #                     [0,0,1,0,0]])/17 #Elliptical
    # kernel = np.array([ [0,0,1,0,0],
    #                     [0,0,1,0,0],
    #                     [1,1,1,1,1],
    #                     [0,0,1,0,0],
    #                     [0,0,1,0,0]])/9 #Cross
    # kernel = np.array([ [0,0,0,0,0],
    #                     [0,1,1,1,0],
    #                     [0,1,1,1,0],
    #                     [0,1,1,1,0],
    #                     [0,0,0,0,0]])/9
    # kernel = np.array([ [1,1,1,1,1],
    #                     [1,0,0,0,1],
    #                     [1,0,0,0,1],
    #                     [1,0,0,0,1],
    #                     [1,1,1,1,1]])/16 #Très flou
    # kernel = np.array([ [1,1,1,1,1,1,1],
    #                     [1,0,0,0,0,0,1],
    #                     [1,0,0,0,0,0,1],
    #                     [1,0,0,0,0,0,1],
    #                     [1,0,0,0,0,0,1],
    #                     [1,0,0,0,0,0,1],
    #                     [1,1,1,1,1,1,1]])/24 #Très flou, chouette avec threshold
    # kernel = np.array([ [5,5,5,5,5],
    #                     [5,3,3,3,5],
    #                     [5,3,1,3,5],
    #                     [5,3,3,3,5],
    #                     [5,5,5,5,5]])/95 #Kind of Dilatation

    #Essayer de trouver des filtres de dilatation, erosion et dilatation-erosion
    return cv2.filter2D(img,-1,kernel)

def movement_detect(now,prev,threshold = 150):
    '''
    Now et Prev sont deux images qui se suivent.
    Elles doivent être en nuance de gris
    '''
    (score,diff) = compare_ssim(now,prev,full=True) #Différence entre les images
    #diff = cv2.absdiff(now,prev) #Autre méthode
    next = (diff*255).astype("uint8") #Utile pour le threshold
    result = cv2.threshold(next,threshold,255,cv2.THRESH_BINARY_INV)
    return result[1]

def draw_contours(img,movement):
    '''
    Dessine les contours des zones en mouvement
    img est l'image sur laquelle on va dessiner.
    Movement est l'image en noir et blanc qui sert à faire la détection de mvt
    '''

    new_img = img.copy()
    blurred = hole_filling(movement,1)
    contours, _ = cv2.findContours(blurred,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(new_img,contours, -1, (0,255,0), 2)
    return new_img

def draw_rect_contours(img,movement,area_size = 700):
    '''
    Dessine des carrés sur les zones en mouvement
    img est l'image sur laquelle on va dessiner.
    Movement est l'image en noir et blanc qui sert à faire la détection de mvt
    area_size permet de définir la taille minimum que le mvt doit avoir pour qu'on
    le considère. Il dépend du contexte
    '''
    new_img = img.copy()
    blurred = hole_filling(movement,1)#Si j'augmente cette valeur, ça augmente la zone et diminue p-ê les lignes
    contours, _ = cv2.findContours(blurred,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #Dessine des rectangles
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        if cv2.contourArea(contour) < area_size: #Dépend du contexte
            continue #On ne s'intéresse qu'aux grandes régions
        cv2.rectangle(new_img,(x,y),(x+w,y+h), (0,255,0), thickness = 2)

    return new_img

def hole_filling(img, passage = 1):
    '''
    Remplit les trous
    Passage = Nombre de fois qu'on va appliquer le kernel croix
    '''
#SOL 1
    # kernel = np.array([ [0,0,1,0,0],
    #                     [0,0,1,0,0],
    #                     [1,1,1,1,1],
    #                     [0,0,1,0,0],
    #                     [0,0,1,0,0]])/9 #Cross
    # output = strided_convolution(img,kernel,1)
    # for i in range(passage-1):
    #     output = strided_convolution(output,kernel,1)
#SOL 2
    output = img.copy()
    for i in range(passage):
        blur = cv2.GaussianBlur(output,(5,5),0)
        _, output = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    return output


def reduce_pixel(img,stride = 1,kernel_size = 3):
    '''
    Réduit le nombre de pixels d'une image à l'aide d'un kernel
    '''
    kernel = np.ones((kernel_size,kernel_size),np.uint8)/(kernel_size**2) #Grand flou
    output = strided_convolution(img, kernel, stride)
    return output

def gray_to_ascii():
    pass
#Idée de fonction, transformer gray scale en ascii

def one_color(img,n):
    '''
    Isole une couleur dans une image RGB
    '''
    new_img = img.copy()
    for i in range(0,3):
        if i == n:
            continue
        new_img[:,:,i] = 0
    return new_img

## ----------------------------------------------------------------------------
''' MAIN CODE '''

# Connects to your computer's default camera
cap = cv2.VideoCapture(0)


# Automatically grab width and height from video feed
# (returns float which we need to convert to integer for later on!)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
ret, frame = cap.read()


while True:

    # Capture frame-by-frame
    prev = frame #J'enregistre la dernière image
    #Fonctionne mieux, l'intervalle entre les deux images est plus grande
    #ret, prev = cap.read()
    ret, frame = cap.read()

    # # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.cvtColor(prev,cv2.COLOR_BGR2GRAY)
    #gray = cv2.cvtColor(movement_detect(frame,prev), cv2.COLOR_BGR2GRAY)


#Filtre lumière
    thresh = cv2.threshold(gray,110,255,cv2.THRESH_BINARY)[1]
    #cv2.imshow('window',result[1])

# Detection de mouvement
    movement = movement_detect(gray,prev_gray)
    contour = draw_contours(frame,movement)
    rect = draw_rect_contours(frame,movement)
    #cv2.imshow('frame',movement)
    cv2.imshow('window',contour)
    cv2.imshow('frame',rect)


#Réduit la taille de l'image
    # result = reduce_pixel(thresh,5,1)
    # cv2.imshow('frame',gray)
    # cv2.imshow('window',result)



    # This command let's us quit with the "q" button on a keyboard.
    # Simply pressing X on the window won't work!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture and destroy the windows
cap.release()
cv2.destroyAllWindows()
