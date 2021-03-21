import cv2
import numpy as np
from numpy.lib.stride_tricks import as_strided
import os
import math
from skelet import resize
import time

'''
Quelques fonctions de traitement d'images qui n'étaient pas nécessaire
dans le détecteur de mouvement

'''


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

def distance_transform(img):
    '''
    Prend une image en noir et blanc en entrée et sort une image en grayscale
    avec la fonction de distance transform


    Tout sera blanc au dessus de 255 pixels
    '''
    #Première étape: fabriquer l'image de sortie avec des 0 dans les pixels noirs

    #On ajoute un pixel autour pour ne pas avoir de problèmes
    height, width = img.shape[:2]
    bottom = img[height-2:height, 0:width]
    mean = cv2.mean(bottom)[0]

    bordersize = 1
    dt = cv2.copyMakeBorder(
    img,
    top=bordersize,
    bottom=bordersize,
    left=bordersize,
    right=bordersize,
    borderType=cv2.BORDER_CONSTANT,
    value=[mean, mean, mean]
    )

    max_val = 0
    #Deuxième étape:  passe de gauche à droite et de haut en bas pour définir la distance
    for y in range(1,height+1):
        for x in range(1,width+1):
            '''
            On test si on est = 0, sinon on prend le minimum entre le haut et la gauche
            et on fait plus un à cette valeur
            '''
            if dt[y, x] != 0:
                dt[y, x] = min(dt[y-1, x], dt[y ,x-1]) + 1

    #Troisième étape:  passe de droite à gauche et de bas en haut pour définir la distance
    for y in range(height, 0, -1):
        for x in range(width, 0, -1):
            neigh = min(dt[y+1, x], dt[y ,x+1])
            if dt[y, x] != 0 and neigh < dt[y, x]:
                dt[y, x] = neigh + 1
            max_val = max(max_val,dt[y,x])


    #ON RETIRE LE PIXEL QU'ON A AJOUTE AU DEBUT
    #au pire on le garde


    #Standardisation
    ratio = int(255/max_val)
    #Le int pourrait poser problème
    dt *= ratio


    return dt








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


# def reduce_pixel(img,stride = 1,kernel_size = 3):
#     '''
#     Réduit le nombre de pixels d'une image à l'aide d'un kernel
#     '''
#     kernel = np.ones((kernel_size,kernel_size),np.uint8)/(kernel_size**2) #Grand flou
#     output = strided_convolution(img, kernel, stride)
#     return output

def get_dominant_color(img):
    '''
    Stolen
    Fonction permettant de connaitre la couleur dominante d'une image

    '''
    pixels = np.float32(img.reshape(-1, 3))

    n_colors = 5
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv2.KMEANS_RANDOM_CENTERS

    _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
    _, counts = np.unique(labels, return_counts=True)

    dominant = palette[np.argmax(counts)]
    return dominant

def get_mean_color(img):
    return img.mean(axis=0).mean(axis=0)


def create_img_set(path, gray = False):
    '''
    Je créé une liste d'image à partir d'images d'un dossier

    '''
    #Gather images
    images = []
    color_set = [0] * 256 #Je créé une liste vide
    for filename in os.listdir(path):
        if not 0 in color_set:
            return color_set

        if ".png" in filename or ".jpg" in filename:
            img = cv2.imread(os.path.join(path,filename))
            if img is not None:
                img = cv2.resize(img,(int(1/ratio),int(1/ratio)))
                if gray == True:
                    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                    mean_col = int(get_mean_color(img))
                    color_set[mean_col] = img
                    print(f"traitement de {filename}")
                    continue
                images.append(img)
    return color_set



def big_pixels(img,ratio = 0.1,images = None):
    '''
    Réduit le nombre de pixels d'une image
    Si je fournis une liste d'images, elle affiche les images en petit
    à la place des pixels
    '''
    img_width = img.shape[1]
    img_height = img.shape[0]
    width = int(img_width * ratio)
    height = int(img_height * ratio)
    dim = (width,height)
    small = cv2.resize(img,dim)
    image = np.zeros(img.shape,np.uint8)

#Ici je remplit mon image en gris
    if len(img.shape) < 3:
        for x in range(width):
            for y in range(height):
                if images == None:
                    cv2.rectangle(image,(int(x/ratio),int(y/ratio)),
                                (int((x+1)/ratio),int((y+1)/ratio)), int(small[y,x]),
                                 thickness = -1)
                else:
                    index = int(small[y,x])
                    if type(images[index]) == type(int(1)):
                        #Je n'ai pas d'images
                        cv2.rectangle(image,(int(x/ratio),int(y/ratio)),
                                    (int((x+1)/ratio),int((y+1)/ratio)),
                                     int(small[y,x]), thickness = -1)
                    else:
                        image[int(y/ratio):int((y+1)/ratio), int(x/ratio):int((x+1)/ratio)] = images[index]
        return image
#Else, l'image est en couleur
    for x in range(width):
        for y in range(height):
            r = int(small[y,x,0])
            g = int(small[y,x,1])
            b = int(small[y,x,2])
            if images == None:
                cv2.rectangle(image,(int(x/ratio),int(y/ratio)),(int((x+1)/ratio),int((y+1)/ratio)), (r,g,b), thickness = -1 )
            #thickness = -1 remplit le rectangle
    return image



#Réduit la taille de l'image
    # result = reduce_pixel(thresh,5,1)
    # cv2.imshow('frame',gray)
    # cv2.imshow('window',result)




#Filtre lumière
    #thresh = cv2.threshold(gray,110,255,cv2.THRESH_BINARY)[1]


#Stolen https://gist.github.com/arifqodari/dfd734cf61b0a4d01fde48f0024d1dc9
def strided_convolution(image, weight, stride):
    im_h, im_w = image.shape
    f_h, f_w = weight.shape

    out_shape = (1 + (im_h - f_h) // stride, 1 + (im_w - f_w) // stride, f_h, f_w)
    out_strides = (image.strides[0] * stride, image.strides[1] * stride, image.strides[0], image.strides[1])
    windows = as_strided(image, shape=out_shape, strides=out_strides)

    return np.tensordot(windows, weight, axes=((2, 3), (0, 1)))



def gray_to_ascii():
    pass
#Idée de fonction, transformer gray scale en ascii
#----------------------------------------------------------------------------
'''MAIN '''
ratio = 0.05

# Connects to your computer's default camera
cap = cv2.VideoCapture(0)

# Automatically grab width and height from video feed
# (returns float which we need to convert to integer for later on!)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


#Le téléchargement (même de 50 images) PREND DU TEMPS

# print("Je démarre le téléchargement")
#img_set = create_img_set("C:\\Users\\william\\OneDrive\\Images\\Pellicule",True)
#img_set = create_img_set("W:\\2020\\Ete\\Chalet TeamSki",True)
# print("J'ai fini")

#Je vais chercher l'image du cercle et je lui donne une taille correcte
circ = cv2.imread("images\\circle.jpg")
circ_big = resize(circ,10,True)
_,circ_thresh = cv2.threshold(circ_big,200,255,cv2.THRESH_BINARY)
circ_dt = distance_transform(circ_thresh)


while True:


    ret, frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    _,thresh = cv2.threshold(gray,125,255,cv2.THRESH_BINARY_INV)
    small = resize(thresh,0.1,False)
    #ici je modifie la taille de mon image


    start = time.time()
    #Ici je calcule la distance
    result = distance_transform(small)
    cv2.imshow('resultat',result)
    cv2.imshow('big',small)

    end = time.time()
    print(f"La fonction DT et l'affichage a prit {end-start} secondes")
    # cv2.imshow('initial',circ_thresh)
    # cv2.imshow('resultat',circ_dt)
    # print(get_dominant_color(result)) #Très long

    # print(get_mean_color(frame))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# When everything done, release the capture and destroy the windows
cap.release()
cv2.destroyAllWindows()
