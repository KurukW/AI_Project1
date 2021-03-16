import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import cv2
from random import randint
from skimage.measure import compare_ssim
from scipy import signal

'''
Amélioration à faire:
- Ajouter système de blob comme "coding train " pour garder la trace de ce qui bouge
- Ajouter une fonction qui sort les coordonnées des zones en mvt.
Ou les images en mvt en fonction de l'utilisation que j'en ai, mais je peux
faire une autre fonction qui va transformer coordonnées en images/

'''
# ----------------------------------------------------------------------------
'''OBJECTS'''
class Zones:
    def __init__(self,img, x,y,w,h):
        '''
        (x,y) coordonnées en haut à gauche
        (w,h) taille du rectangle
        '''
        self.img = img[x:x+w, y:y+h]

    def show(self):
        cv2.imshow('myObject',self.img)

# ----------------------------------------------------------------------------
'''FONCTIONS '''
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

def show_zones(zones):
    '''
    zones est une liste avec les objets zone

    '''
    if len(zones) >= 0:
        cv2.imshow('Zones',zones[0].img)



def get_contours(movement):
    blurred = hole_filling(movement)
    contours, _ = cv2.findContours(blurred,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def extract_movement_zones(img,contours):
    '''
SEMBLABLE A L'OBJET ZONES - CA NE FONCTIONNE PAS
    img = image dans laquelle on va découper des zones en mouvement
    contours= liste des coordonnées des zones en mouvement
    '''
    zones = [] #sections d'images en mouvement
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        new_zone = img[x:x+w, y:y+h]
        zones.append(new_zone)
    return zones



def write_on_image(img,text, color = (255,255,255)):
    '''
    Ecrit du texte sur une image
    L'image est un objet passé par ref et non par valeur
    Quand on modifie l'image, on modifie l'original

    '''
    #Ecrit en bas de l'image le nombre de contours qu'il y a dedans
    #Bout de code temporaire, c'est pour les tests
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10,height-15)
    fontScale              = 1
    fontColor              = color
    lineType               = 2

    cv2.putText(img,text,
        bottomLeftCornerOfText,
        font,
        fontScale,
        fontColor,
        lineType)


def draw_contours(img,movement):
    '''
    Dessine les contours des zones en mouvement
    img est l'image sur laquelle on va dessiner.
    Movement est l'image en noir et blanc qui sert à faire la détection de mvt
    '''
    new_img = img.copy()
    contours = get_contours(movement)

    cv2.drawContours(new_img,contours, -1, (0,255,0), 2)
    write_on_image(new_img,str(len(contours)))
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
    contours = get_contours(movement)

    #Dessine des rectangles
    n_contours = 0
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        if cv2.contourArea(contour) < area_size: #Dépend du contexte
            continue #On ne s'intéresse qu'aux grandes régions
        cv2.rectangle(new_img,(x,y),(x+w,y+h), (0,255,0), thickness = 2)
        n_contours += 1
    write_on_image(new_img,str(n_contours))
    return new_img

def hole_filling(img,kernel_size = 15, passage = 1):
    '''
    Remplit les trous
    Passage = Nombre de fois qu'on va appliquer le flou
    Kernel_size = taille du kernel. Pour un kernel_size élevé, le passage peut-être petit
    Pour un kernel_size petit, il faut plusieurs passages pour un résultat proche
    '''
    output = img.copy()
    for i in range(0,passage):
        blur = cv2.GaussianBlur(output,(kernel_size, kernel_size),0)
        #Je trouve que 15 c'est pas trop large et ça réduit vraiment le Nombre
        #de contours différents
        #J'ai vite fait lu un forum et Gaussian c'est mieux ici car plus rapide
        #Et arrondi les bords en plus de réduire le bruit.
        #blur = cv2.medianBlur(output,5,0)
        _, output = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    return output
## ----------------------------------------------------------------------------
''' MAIN CODE '''
#On peut inclure le movement detection dans un autre code et avoir
#Les fonctions sans le code main
if __name__ == "__main__":
    # Connects to your computer's default camera
    cap = cv2.VideoCapture(0)

    # Automatically grab width and height from video feed
    # (returns float which we need to convert to integer for later on!)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    #Je le met une fois dehors pour avoir une première image à copier dans prev
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    while True:

        # Capture frame-by-frame
        prev_gray = gray #J'enregistre la dernière image
        ret, frame = cap.read()

        # # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detection de mouvement
        movement = movement_detect(gray,prev_gray) #Noir et blanc avec mvt
        contour = draw_contours(frame,movement) # Contour du mvt sur rgb
        rect = draw_rect_contours(frame,movement,700) #Carré sur mvt
        cv2.imshow('Contours',contour)
        cv2.imshow('Rectangles',rect)
        #cv2.imshow('Gris',gray)
        #zones = extract_movement_zones(frame,get_contours(movement))
        #Ne fonctionne pas

        # This command let's us quit with the "q" button on a keyboard.
        # Simply pressing X on the window won't work!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture and destroy the windows
    cap.release()
    cv2.destroyAllWindows()
