from Movement_detection import *
import cv2
import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
import scipy.spatial
import matplotlib.pyplot as plt

'''

J'ai juste fait quelques tests, rien de concluant pour l'instant


'''



def blur_thresh(img,passe = 1,kernel_size = 3,thresh = 200):
    blur = img.copy()
    for i in range(passe):
        blur = cv2.GaussianBlur(blur,(kernel_size,kernel_size),0)
        _,blur = cv2.threshold(blur,thresh,255,cv2.THRESH_BINARY)
    return blur


def resize(img,ratio = 2,toGray = True):
    width = int(img.shape[1] * ratio)
    height = int(img.shape[0] * ratio)
    img_big = cv2.resize(img,(width,height),cv2.INTER_LINEAR)
    img_gray = cv2.cvtColor(img_big, cv2.COLOR_RGB2GRAY)
    return img_big, img_gray

def resize_and_draw_contours(img,ratio = 2):
    '''
    Prend les petites images que j'ai créé dans le dossier images
    les aggrandit et dessine les contours dessus

    '''
    img_big, img_gray = resize(img,ratio,True)
    img_cont = get_contours(img_gray) #get array with contours coordinates
    img_output = cv2.drawContours(img_big,img_cont,-1,(0,255,0))
    img_contours = np.zeros((img_big.shape[1],img_big.shape[0]),dtype=np.uint8)
    cv2.drawContours(img_contours,img_cont,-1,(255,255,255))
    return img_output, img_contours,img_cont

def draw_contours_points(contours,shape=(480,480)):
    '''Je dessine chaque trait séparément pour mieux comprendre le bazar'''

    new_cont = np.zeros(shape,dtype=np.uint8)
    new_shape = new_cont.copy()
    for shape in contours:
        for e,contour in enumerate(shape):
            pos1 = tuple(contour[0])
            pos2 = (200,200)

            if e < (len(shape)-1):
                pos2 = tuple(shape[e+1][0])
            else:
                pos2 = tuple(shape[0][0])

            center = (int((pos1[0]+pos2[0])/2),int((pos1[1]+pos2[1])/2))
            cv2.line(new_cont,pos1,pos2,(255,255,255),thickness=1)
            cv2.circle(new_shape,center,radius = 0,color=(255,255,255),thickness = -1) #-1 to fill
    return new_cont,new_shape



def get_voronoi(img,ratio = 2,step = 3):
    start = time.time()
    #Voronoi est une librairie qui va calculer voronoi pour nous
    #Je traduis mes points manuellement

    #Eroder mon image pour avoir une version plus petite pour avoir moins de
    # mauvais traits
    blurred = blur_thresh(img)
    _,_, blurred_contours = resize_and_draw_contours(blurred,ratio)
    img,img_contours_only,img_contours = resize_and_draw_contours(img,ratio)

    points = [elt[0] for elt in img_contours[0]]
    vor = Voronoi(points)
    fig = voronoi_plot_2d(vor)
    # plt.show()
    # print(scipy.spatial.__file__) #Permet de trouver la localisation sur son pc

    #Autre solution: dessin sur un cv2
    vd = np.zeros(img.shape)#voronoi diagram

    # cv2.polylines(vd,vor.vertices,True,(0,255,255))
    vor_points = vor.vertices #Points du futur squelette
    vor_indices_dirty = vor.ridge_vertices #Indice indiquant comment dessiner les droites
    vor_indices = []
    #Nettoyage de vor_indices, on a pas besoin des élmts avec -1
    for segment in vor_indices_dirty:
        good = True
        for indice in segment:
            if indice == -1:
                good = False
                break
        if good:
            vor_indices.append(segment)

    #comme étape 1 mais sans couleurs
    if step == 0:
        for segment in vor_indices:
            pt1 = (int(vor_points[segment[0]][0]), int(vor_points[segment[0]][1]))
            pt2 = (int(vor_points[segment[1]][0]), int(vor_points[segment[1]][1]))
            white = (255,255,255)
            cv2.line(vd,pt1,pt2,white,1)

    #Première étape, il y a tous les traites de Voronoi
    if step == 1:
        for segment in vor_indices:
            pt1 = (int(vor_points[segment[0]][0]), int(vor_points[segment[0]][1]))
            pt2 = (int(vor_points[segment[1]][0]), int(vor_points[segment[1]][1]))
            for contour in img_contours:
                if cv2.pointPolygonTest(contour,pt1,False)>0 and cv2.pointPolygonTest(contour,pt2,False)>0:
                    color = (0,255,0)
                    break
                else:
                    color = (0,0,255)
            # white = (255,255,255)
            cv2.line(vd,pt1,pt2,color,1)

    #Deuxième étape, il y a seulement les traits à l'intérieur de la forme
    if step == 2:
        for segment in vor_indices:
            pt1 = (int(vor_points[segment[0]][0]), int(vor_points[segment[0]][1]))
            pt2 = (int(vor_points[segment[1]][0]), int(vor_points[segment[1]][1]))
            for contour in img_contours:
                if cv2.pointPolygonTest(contour,pt1,False)>0 and cv2.pointPolygonTest(contour,pt2,False)>0:
                    color = (0,255,0)
                    cv2.line(vd,pt1,pt2,color,1)
                    break

    #Troisième étape, il y a seulement les traits à l'intérieur de la forme
    if step == 3:
        for segment in vor_indices:
            pt1 = (int(vor_points[segment[0]][0]), int(vor_points[segment[0]][1]))
            pt2 = (int(vor_points[segment[1]][0]), int(vor_points[segment[1]][1]))
            for contour in blurred_contours:
                if cv2.pointPolygonTest(contour,pt1,False)>0 and cv2.pointPolygonTest(contour,pt2,False)>0:
                    color = (0,255,0)
                    cv2.line(vd,pt1,pt2,color,1)
                    break




    end = time.time()
    print(f"Voronoi a prit {end-start} secondes")
    return vd

#----------------------------------------------------------------------------
'''Main code '''
if __name__ == '__main__':
    rect = cv2.imread("images\\rectangle.jpg")
    circ = cv2.imread("images\\circle.jpg")
    plant = cv2.imread("images\\plant.jpg")
    hand_init = cv2.imread("images\\hand.png")

    _,plant = cv2.threshold(plant,150,255,cv2.THRESH_BINARY_INV)
    _,hand = cv2.threshold(hand_init,150,255,cv2.THRESH_BINARY_INV)

    plant, plant_contour_only, plant_contours = resize_and_draw_contours(plant,1)
    hand, hand_contour_only, hand_contours = resize_and_draw_contours(hand,1)
    rect_big, rect_contour_only, rect_contours = resize_and_draw_contours(rect,10)
    circ_big, circ_cont, _ = resize_and_draw_contours(circ,10)

    new_cont,new_shape = draw_contours_points(rect_contours)

    blur = blur_thresh(plant,1,15,150)


    vd = get_voronoi(hand,1)
    vd0 = get_voronoi(hand,1,0)
    vd1 = get_voronoi(hand,1,1)
    vd2 = get_voronoi(hand,1,2)
    cv2.imwrite('contours.jpg',hand_contour_only)

    # cv2.imwrite('vd0.jpg',vd0)
    # cv2.imwrite('vd1.jpg',vd1)
    # cv2.imwrite('vd2.jpg',vd2)
    # cv2.imwrite('vd3.jpg',vd)








    while True:
        # if cv2.waitKey(1) & 0xFF == ord('n'):
        #     blur = blur_thresh(blur,1,15,150)
        #


        cv2.imshow('shape_rect',hand_init)
        cv2.imshow('voronoi',vd)
        # cv2.imshow('contours',hand_contour_only)
        cv2.imshow('vd0',vd0)
        cv2.imshow('vd1',vd1)
        cv2.imshow('vd2',vd2)



        # cv2.imshow('blurred',blur)

        # cv2.imshow('cont_rect',rect_cont)
        # cv2.imshow('cont_circ',circ_cont)
        # cv2.imshow('shape_circ',circ_big)
        # cv2.imshow('new',new_cont)
        # cv2.imshow('points',new_shape)



        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # When everything done, release the capture and destroy the windows
    # cap.release() #Pour la caméra
    cv2.destroyAllWindows()






'''
ALGORITHME DE SQUELETISATION P97 du pdf ou 74 des feuilles

Explication de chaque ligne que j'ai compris

'Vsites := Subsample(C)'
Subsample est un échantillonage.
(au pire on échantillone pas, mais c'est lourd)
On réduit le nombre de points du contour pour avoir un calcul de voronoi
plus facile.
Tous les points du contour qui reste sont des sites de Voronoi


'V := VoronoiTesselation(Vsites);'
V est l'ensemble des polygones de voronoi fabriqué à partir de l'ensemble
des points Vsites

IL FAUT LIRE B.4 GRAPH MATCHING

'foreach sij in V do'
sij est l'intersection entre deux et sltm deux polygones de voronoi.
sij est donc un segment de droite.

'(alphaij,wij) := EndPoints(sij);'
à priori, alpha(ij) et w(ij) sont les points aux extrémités du segment sij





Fonction DT (déjà construite et pas nécessaire finalement)
Meilleure solution : http://vision.cs.utexas.edu/378h-fall2015/slides/lecture4.pdf
page 4:
Deux passes, afin de définir la distance.




EXPLICATIONS DE SCIPY.SPATIAL.VORONOI

vor.vertices: Liste des points qui sont les extrémités de segments de Voronoi
vor.ridge_vertices: Défini chaque segments, les deux points sont les indices des vertices.
-1 signifie que le segment n'est pas fermé et qu'il va à l'infini.
Cela ne nous intéresse pas, on veut seulement les segments fermés



'''
