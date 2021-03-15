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


'''
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
    Passage = Nombre de fois qu'on va appliquer le flou
    '''
    output = img.copy()
    for i in range(passage):
        blur = cv2.GaussianBlur(output,(5,5),0)
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
        #gray = cv2.cvtColor(movement_detect(frame,prev), cv2.COLOR_BGR2GRAY)

    # Detection de mouvement
        movement = movement_detect(gray,prev_gray) #Noir et blanc avec mvt
        contour = draw_contours(frame,movement) # Contour du mvt sur rgb
        rect = draw_rect_contours(frame,movement,700) #Carré sur mvt
        cv2.imshow('window',contour)
        cv2.imshow('frame',rect)




        # This command let's us quit with the "q" button on a keyboard.
        # Simply pressing X on the window won't work!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture and destroy the windows
    cap.release()
    cv2.destroyAllWindows()
