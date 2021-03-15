import cv2
import numpy as np
from numpy.lib.stride_tricks import as_strided

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


def reduce_pixel(img,stride = 1,kernel_size = 3):
    '''
    Réduit le nombre de pixels d'une image à l'aide d'un kernel
    '''
    kernel = np.ones((kernel_size,kernel_size),np.uint8)/(kernel_size**2) #Grand flou
    output = strided_convolution(img, kernel, stride)
    return output

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
