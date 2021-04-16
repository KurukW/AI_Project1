l '''
Ce programme n'a servi qu'une seule fois pour réparer des mauvaises vidéos.
Avec le nouveau programme de film. Ce n'est plus nécessaire


Le but du programme est d'ajouter ou supprimer des images aux vidéos qui n'en ont pas le bon nombre
On veut 61 images, certaines en ont 59, 58, 60 ou 62
'''
import cv2
import pandas as pd
import matplotlib.pyplot as plt


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
            #frame_RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


            #Récupère seulement certaines images
            t_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            modulo = t_ms % ecart_voulu
            if modulo < ecart_initial:
                #Isoler les images
                imgs.append(frame)

        else: #Va jusqu'au bout de la vidéo
            break
    else:
        print("Le fichier n'a pas pu être ouvert")
    cap.release()

    return imgs


def add_img(imgs,n_to_add):
    new_imgs = imgs.copy()
    for i in range(0,n_to_add):
        new_imgs.append(imgs[-1])
    return new_imgs

def del_img(imgs,img_to_keep):
    return [imgs[i] for i in range(img_to_keep)]

def imgs_to_video(imgs,video_name,folder):
    width = len(imgs[0][0])
    height = len(imgs[0])
    file_name = folder + "\\" + video_name
    out = cv2.VideoWriter(file_name,cv2.VideoWriter_fourcc(*'DIVX'), 25, (width,height))

    for img in imgs:
        out.write(img)
    out.release()


if __name__ == '__main__':
    labels_csv = pd.read_csv("labels.csv")
    labels_val = list(labels_csv.values)

    for label,video_name in labels_val:
        file_name = "Videos\\" + video_name
        imgs = get_imgs_from_path(file_name)
        imgs_len = len(imgs)
        if imgs_len == 61:
            continue
        #Les cas à problèmes
        print(imgs_len)
        img_needed = 61 - imgs_len
        if img_needed > 0:
            new_imgs = add_img(imgs,img_needed)
        else:
            new_imgs = del_img(imgs,61)
        imgs_to_video(new_imgs,video_name,'Videos_unbroken')
