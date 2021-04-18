import cv2
import os
import time
import pandas as pd

'''
Ce programme va nous permettre de fabriquer des images
'''


label = "tests" #LE PLUS IMPORTANT

duree_s = 1 #Durée du timer avant enregistrement
video_name = "" #Si c'est vide, le numéro est incrémenté à chaque fois: "video_X.avi"
folder = "Videos"
framerate = 25
#Framerate du rendu final. Cela ne définit pas le nombre d'images qu'on lui donne.
#Si on a un framerate de 20 et la vidéo de 10. ça veut dire qu'une seconde d'enregistrement
# donne 20 images et donc 2 secondes de vidéo


#---------------------------------------------------------------------
'''
Objets
'''
class Video_saving:
    def __init__(self,framerate,width,height,label,folder = "",video_name = "",n_frames = 60):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        file_name,video_name = self.create_name(folder,video_name,label)
        self.out = cv2.VideoWriter(file_name,fourcc, framerate,(width,height))
        self.last = time.time()
        self.frame = 0
        self.n_frames = n_frames
        write_label(label,video_name)

    def update(self,frame):
        start_again = True
        now = time.time()
        ecart = 1.0/framerate
        time_from_last = now - self.last
        if time_from_last >= ecart:
            self.last = now - ecart
            if self.frame <= self.n_frames:
                self.out.write(frame)
                print(f"Je suis à {self.frame} images sur {self.n_frames}",end="\r")

                self.frame += 1
            else:
                #print() # Passe à la ligne suivante, je ne le met pas pour avoir "j'ai fini 60 images sur 60", ça donne bien aussi
                start_again = False
        return (start_again, self.frame)

    def create_name(self,folder,video_name,label):
        #Compte le nombre de fichier (ancienne méthode)
        files = os.listdir("Videos")
        file_count = len(files)


        #Num = last+1 (nouvelle méthode)
        labels = pd.read_csv("labels.csv")
        next_index = max([int(file_name[6:-4]) for _, file_name in labels.values]) + 1 #Ne pas oublier le +1
        #Récupère l'index maximum actuel dans le fichier labels.csv

        #Fabrication du nom de la vidéo
        if video_name == "": #Nom automatique
            #video_name = f"video_{str(file_count)}.avi"
            video_name = f"video_{next_index}.avi"
        if folder == "":
            file_name = video_name
        else:
            file_name = folder + "\\" + video_name


        print(f"Cette vidéo avec le label '{label}' va s'enregistrer sous le nom : {video_name}")
        return file_name, video_name

    def __del__(self): #Quand l'objet est supprimé, on ferme le fichier
        self.out.release()










#-------------------------------------------------------------------------------
'''Fonctions '''
def write_on_image(img,text, color = (255,255,255),position = "bottom left"):
    '''
    Ecrit du texte sur une image
    L'image est un objet passé par ref et non par valeur
    Quand on modifie l'image, on modifie l'original

    '''
    #Placement
    if position == "bottom left":
        bottomLeftCornerOfText = (10,height-15)
    if position == "top left":
        bottomLeftCornerOfText = (10, 25)

    #Ecrit en bas de l'image le nombre de contours qu'il y a dedans
    #Bout de code temporaire, c'est pour les tests
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    fontScale              = 0.8
    fontColor              = color
    lineType               = 2

    cv2.putText(img,text,
        bottomLeftCornerOfText,
        font,
        fontScale,
        fontColor,
        lineType)

def write_label(label,file_name):
    #Génère les labels la toute première fois
    # labels = open("DATA\\labels.csv", "w")
    # labels.write("label, file_name")
    labels = open("labels.csv","a")
    labels.write("\n" + label + "," + file_name) #Ajoute une nouvelle info sur une nouvelle ligne
    labels.close()
    #Parfois le texte s'écrit que la même ligne alors j'ai mis \n pour être sur

#-------------------------------------------------------------------------------
'''MAIN CODE '''
if __name__ == "__main__":
    global video_to_saved
    # Connects to your computer's default camera
    cap = cv2.VideoCapture(0)

    soon_saving = False
    label_num = 0

    labels_file = pd.read_csv('labels_list.csv')
    labels = labels_file.values
    # Automatically grab width and height from video feed
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')



    while True:
        label = labels[label_num][0]
        #Ecriture par défaut
        s = label
        color = (255,255,255)


        ret, frame = cap.read()
        # Pendant 1 mili-seconde, on attend et on regarde si on a appuyé sur une touche
        key = cv2.waitKey(1)

        if key == ord('p'):
            soon_saving = True
            start_soon = time.time()
        if key == ord('n'):
            #Pas d'anti rebond ou spam mais ça a l'air d'être bien géré seul
            label_num += 1
            if (label_num >= len(labels_file)):
                label_num = 0


        #Timer avant enregistrement
        if soon_saving:
            now = time.time()
            time_left = duree_s - (now - start_soon)
            time_left_f = "{:.2f}".format(time_left)
            s = time_left_f
            if time_left <= 0:
                soon_saving = False
                my_video = Video_saving(framerate,width,height,label,folder = folder,n_frames = 60)


        try:
            #Je ne rentre ici que si j'ai un objet, sinon ça fait une erreur et on exécute pas le code
            start_again, n_frame = my_video.update(frame)
            if start_again:
                s = f"Enregistrement, frame {n_frame}"
                color = (0,0,255)
            else:
                print("j'ai fini")
                del my_video

        except NameError:
            pass

        #à partir d'ici, les modifications sur frame n'auront pas d'effet sur l'enregistrement
        #indique à l'image si on enregistre (non visible sur la vidéo)
        write_on_image(frame,s,color,"bottom left")
        write_on_image(frame,"P to save; N to switch label; Q to quit",(255,255,255),"top left")

        cv2.imshow('frame',frame)

        #Quitter
        if key == 113: # 113 = ord('q')
            break
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
    # When everything done, release the capture and destroy the windows
    cap.release()
    cv2.destroyAllWindows()

'''
    #Ferme le fichier vidéo
    out.release()
    #Créer le label pour la nouvelle vidéo
    if video_to_saved:
        write_label(label,video_name)
    else:
        #Il faut supprimer la vidéo qu'on vient de faire
        os.remove(file_name)
'''
