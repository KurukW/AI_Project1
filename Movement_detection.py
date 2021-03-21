import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import cv2
from random import randint
from skimage.measure import compare_ssim
from scipy import signal
import time

'''
Amélioration à faire:
- Ajouter système de blob comme "coding train " pour garder la trace de ce qui bouge
- Ajouter une fonction qui sort les coordonnées des zones en mvt.
Ou les images en mvt en fonction de l'utilisation que j'en ai, mais je peux
faire une autre fonction qui va transformer coordonnées en images/



NOTES:
Pour l'instant, il y a concurrence entre les objets et les rectangles.
En effet, j'ai écrit les rectangles avant et maintenant je travaille pour
transformer tout ça en objet.
(le but étant de pouvoir transférer les images qui seront stocké dans les objets)
Tracasss ça va fonctionner


Je dois bien travailler la persévérance.
Si je détecte un mouvement, il faut l'identifier et le garder.
Si je détecte une main, je dois la garder et l'identifier tant qu'elle est à l'écran.
Même si je ne la trouve plus, je sais où elle était et je peux essayer de la retrouver
avec le mouvement
'''
# ----------------------------------------------------------------------------
'''OBJECTS'''
proximity_thresh = 10
min_area_thresh = 100
#Si la distance entre deux zones est inférieures à cette valeur, je les fusionne


class Zones:
    def __init__(self, x, y, w, h):
        '''
        (x,y) coordonnées en haut à gauche
        (w,h) taille du rectangle
        '''
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def create_my_img(self, img):
        self.img = img[self.y:self.y+self.h, self.x:self.x+self.w]

    def dist_to_me(self, x, y, w, h):
        '''
        Mesure la distance entre notre côtés et le côté de l'autre
        Je vais calculer la distance entre les centres
        Ensuite avec la taille de mes zones je peux définir la distance entre les bordures
        '''

        #Je t'ai fait des grands noms exprès pour que ce soit plus clair
        me_center_x = self.x + self.w/2.0
        me_center_y = self.y + self.h/2.0
        other_center_x = x + w/2.0
        other_center_y = y + h/2.0

        #Sol 1 en calculant la racine
        center_dist = sqrt_dist(me_center_x, me_center_y, other_center_x, other_center_y)
        center_edge_both = np.sqrt((self.w/2.0 + w/2.0)**2 + (self.h/2.0 + h/2.0)**2)
        edge_dist = center_dist - center_edge_both
        #Sol 2 sans calculer la racine
        #Je n'ai aps encore trouvé
        return edge_dist


        #Sol 2 sans calculer la racine
        #Je n'ai aps encore trouvé

    def is_close_to_me(self, x, y, w, h):
        '''
        Fonction obselete
        Compare notre position avec un nouvel objet pour savoir si on doit
        l'ajouter à cet objet ou pas
        On va comparer les distances au carré pour ne pas calculer la racine
        '''
        if self.dist_to_me(x,y,w,h) < proximity_thresh:
            return True
        return False

    def add_close(self, x, y, w, h):
        #Modifications de mes coordonnées
        self.x = min(x, self.x)
        self.y = min(y, self.y)
        maxx = max(self.x + self.w, x + w)
        maxy = max(self.y + self.h, y + h)
        self.w = maxx - self.x
        self.h = maxy - self.y
        #Je fusionne les deux zones

    def draw_on_image(self,img,color = (0,255,255),thickness = 2):
        cv2.rectangle(img,(self.x,self.y),(self.x+self.w,self.y+self.h), color = color, thickness = thickness)

    def show(self,title):
        cv2.imshow(title,self.img)

# ----------------------------------------------------------------------------
'''FONCTIONS '''
def movement_detect(now,prev,threshold = 150,sol = 1):
    '''
    Now et Prev sont deux images qui se suivent.
    (Elles doivent être en nuance de gris)
    Je renvoie une image en noir et blanc avec les images
    '''
    #Sol 1
    if sol == 1:
        (score,diff) = compare_ssim(now,prev,full=True) #Différence entre les images
        next = (diff*255).astype("uint8") #Utile pour le threshold
        result = cv2.threshold(next,threshold,255,cv2.THRESH_BINARY_INV)
        return result[1]
    #Sol 2
    diff = cv2.absdiff(now,prev)
    result = cv2.threshold(diff,threshold,255,cv2.THRESH_BINARY)
    output = cv2.cvtColor(result[1],cv2.COLOR_BGR2GRAY)
    return output

def sqrt_dist(x1, y1, x2 ,y2):
    '''
    Calcule la distance réelle entre deux pixels
    '''
    return np.sqrt((x1-x2)**2 + (y1-y2)**2)


# J'ai commenté ces deux fonctions qui sont écrites trop tôt
#Je ne dois pas ouvrir plein de fenêtres différntes, je dois trouver
#un moyen de mettre toutes les images dans une même fenêtre.
#Afin d'avoir une fenêtre fixe

# def create_zones_img(img,zones):
#     '''
#     Créer l'image de chaque zone
#     '''
#     if len(zones) == 0:
#         return
#     for zone in zones:
#         zone.create_my_img(img)
# def show_zones(zones):
#     '''
#     Sort une fenêtre par zone
#     '''
#     if len(zones) == 0:
#         return
#     for e,zone in enumerate(zones):
#         zone.show(f"Numéro {e}")

#
# def show_concatene_img_zones(zones):
# '''
# Fonction écrites avec les pieds
# Ne fonctionne pas car je dois concaténer des img qui ont la même taille
# '''
#     size = int(np.sqrt(len(zones)))
#     #int arrondi en dessous donc pas de soucis
#     output_img = zones[0].img
#     for i in range(1,len(zones)):
#         output_img = np.concatenate((output_img, zones[i].img),axis = 1)
#
#     return output_img




def get_contours(movement):
    '''
    L'image doit être blanche et le fond noir pour trouver les contours
    '''
    blurred = hole_filling(movement)
    contours, _ = cv2.findContours(blurred,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours

# def extract_movement_zones(img,contours):
#     '''
# SEMBLABLE A L'OBJET ZONES
#     img = image dans laquelle on va découper des zones en mouvement
#     contours= liste des coordonnées des zones en mouvement
#     '''
#     zones = [] #sections d'images en mouvement
#     for contour in contours:
#         (x, y, w, h) = cv2.boundingRect(contour)
#         new_zone = img[y:y+h, x:x+w]
#         zones.append(new_zone)
#         cv2.imshow('Mouvement',new_zone)
#     return zones

def show_movement_zones(img,contours):
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        #print(x,y,w,h)
        if (w*h) > 1500: #Taille minimale
            new_zone = img[y:y+h, x:x+w]
            #x et y sont dans le sens inverse d'ailleurs, ce n'est pas logique
            cv2.imshow('Mouvement',new_zone)


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
        
def draw_contours(img,contours):
    '''
    Dessine les contours des zones en mouvement
    img est l'image sur laquelle on va dessiner.
    Movement est l'image en noir et blanc qui sert à faire la détection de mvt
    '''
    new_img = img.copy()

    # -1 signifie qu'on va dessiner tous les contours
    #Si on met une autre valeur, c'est l'index du contour à dessiner
    cv2.drawContours(new_img,contours, -1, (0,255,0), 2)
    write_on_image(new_img,str(len(contours)))
    return new_img

def draw_rect_contours(img,contours,area_size = 700):
    '''
    Dessine des carrés sur les zones en mouvement
    img est l'image sur laquelle on va dessiner.
    Movement est l'image en noir et blanc qui sert à faire la détection de mvt
    area_size permet de définir la taille minimum que le mvt doit avoir pour qu'on
    le considère. Il dépend du contexte
    '''
    new_img = img.copy()
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

def generate_zones(contours):
    zones = []
    #Analyse et dessins des contours par la méthode objet
    for contour in contours:
        if cv2.contourArea(contour) < min_area_thresh: #Dépend du contexte
            continue #On ne s'intéresse qu'aux grandes régions
        (x, y, w, h) = cv2.boundingRect(contour)
        if len(zones) == 0:
            #Pas d'objets, j'ajoute le premier
            zones.append(Zones(x,y,w,h))

        #Je compare le nouveau contour aux objets existants
        shortest_dist = width * height #Je suis certain d'être supérieur au max
        closest_zone = 0
        for existing_zone in zones:
            #Je cherche la zone la plus proche
            existing_zone_dist = existing_zone.dist_to_me(x, y, w, h)
            if existing_zone_dist < shortest_dist:
                shortest_dist = existing_zone_dist
                closest_zone = existing_zone

        if shortest_dist < proximity_thresh:
            closest_zone.add_close(x, y, w, h)
        else:
        #Je créé une nouvelle zone
            zones.append(Zones(x,y,w,h))
    return zones

def draw_all_zones(img,zones):
    img_by_zones = img.copy()
    for zone in zones:
        zone.draw_on_image(img_by_zones)
    write_on_image(img_by_zones,str(len(zones)))
    return img_by_zones

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
        prev = frame
        ret, frame = cap.read()
        # # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detection de mouvement
        # movement = movement_detect(gray,prev_gray) #Noir et blanc avec mvt
        #contours = get_contours(movement)
        #
        # contour = draw_contours(frame,contours) # Contour du mvt sur rgb
        # rect = draw_rect_contours(frame,contours,700) #Carré sur mvt
        # cv2.imshow('Contours',contour)
        # cv2.imshow('Rectangles',rect)

    ##Différence entre les deux fonctions d'analyse de Mouvement
        movement1 = movement_detect(gray,prev_gray,150,1)
        movement2 = movement_detect(frame,prev,20,2)
        cv2.imshow('Sol 1', movement1)
        cv2.imshow('Sol 2', movement2)
        #contours = get_contours(movement1)
        #print(contours[0])

        #show_movement_zones(frame,contours)

        # zones = generate_zones(contours)
        # #Toutes mes zones sont créées, mtn je dois les dessiner sur l'image et l'afficher
        # img_by_zones = draw_all_zones(frame, zones)
        # cv2.imshow('Dessin',img_by_zones)
        # #

        #Pas fonctionnel
        # create_zones_img(frame,zones)
        # concatene = show_concatene_img_zones(zones)
        # cv2.imshow('Le Bordel',concatene)


        # This command let's us quit with the "q" button on a keyboard.
        # Simply pressing X on the window won't work!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # When everything done, release the capture and destroy the windows
    cap.release()
    cv2.destroyAllWindows()
