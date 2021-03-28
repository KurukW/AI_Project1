import cv2
import os
import time
'''
Ce programme va nous permettre de fabriquer des images
Pour démarrer l'enregistrement, appuyer sur "r"
Pour arrêter l'enregistrement, appuyer sur "t"

Si on démarre le programme sans enregistrer de vidéo, on va créer un fichier vidéo vide, c'est mal.
'''


label = "Video de nous" #LE PLUS IMPORTANT

duree_s = 3 #Durée de la vidéo quand on enregistre avec k
video_name = "" #Si c'est vide, le numéro est incrémenté à chaque fois: "video_X.avi"
folder = "DATA\\Videos"
framerate = 25
#Framerate du rendu final. Cela ne définit pas le nombre d'images qu'on lui donne.
#Si on a un framerate de 20 et la vidéo de 10. ça veut dire qu'une seconde d'enregistrement
# donne 20 images et donc 2 secondes de vidéo


#-------------------------------------------------------------------------------
'''Fonctions '''
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

def write_label(label,file_name):
    #Génère les labels la toute première fois
    # labels = open("DATA\\labels.csv", "w")
    # labels.write("label, file_name")
    labels = open("DATA\\labels.csv","a")
    labels.write("\n" + label + "," + file_name) #Ajoute une nouvelle info sur une nouvelle ligne
    labels.close()
    #Parfois le texte s'écrit que la même ligne alors j'ai mis \n pour être sur

#-------------------------------------------------------------------------------
'''MAIN CODE '''
if __name__ == "__main__":
    global video_to_saved
    # Connects to your computer's default camera
    cap = cv2.VideoCapture(0)

    # Automatically grab width and height from video feed
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    saving = False
    soon_saving = False
    video_to_saved = False
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    #Compte le nombre de fichier
    files = os.listdir("DATA\\Videos")
    file_count = len(files)

    #Fabrication du nom de la vidéo
    if video_name == "": #Nom automatique
        video_name = f"video_{str(file_count)}.avi"

    if folder == "":
        file_name = video_name
    else:
        file_name = folder + "\\" + video_name

    #Création du fichier de sortie
    out = cv2.VideoWriter(file_name,fourcc, framerate,(width,height))



    while True:



        ret, frame = cap.read()
        # Pendant 1 seconde, on attend et on regarde si on a appuyé sur une touche
        key = cv2.waitKey(1)

        #Défini si on enregistre ou pas
        if key == ord('r'):
            saving = 1 #Démarre enregistrement
        if key == ord('t'):
            saving = 0 #Arrête enregistrement
        if key == ord('k'):
            soon_saving = True
            start_soon = time.time() #Démarre timer avant enregistrement chronométré




        #Enregistre
        if saving == 1: #Enregistrement indéfini
            s = "Enregistrement "
            video_to_saved = True
            color = (0,0,255)
            out.write(frame)
        elif saving == 2: #Enregistrement défini
            s = "Enregistrement "
            video_to_saved = True
            out.write(frame)

            #Gestion du timer
            now = time.time()
            time_left = duree_s - (now - start)
            time_left_f = "{:.2f}".format(time_left)
            s += time_left_f
            color = (0,0,255)
            if time_left <= 0:
                saving = 0
        else: #Pas enregistrement
            s = "Nothing"
            color = (255,255,255)

        #Timer avant enregistrement
        if soon_saving:
            now = time.time()
            time_left = 3 - (now - start_soon)
            time_left_f = "{:.2f}".format(time_left)
            s = time_left_f
            if time_left <= 0:
                soon_saving = False
                saving = 2
                start = time.time()



        #à partir d'ici, les modifications sur frame n'auront pas d'effet sur l'enregistrement

        #indique à l'image si on enregistre (non visible sur la vidéo)
        write_on_image(frame,s,color)

        cv2.imshow('frame',frame)

        #Quitter
        if key == 113: # 113 = ord('q')
            break
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
    # When everything done, release the capture and destroy the windows
    cap.release()
    cv2.destroyAllWindows()


    #Ferme le fichier vidéo
    out.release()
    #Créer le label pour la nouvelle vidéo
    if video_to_saved:
        write_label(label,video_name)
    else:
        #Il faut supprimer la vidéo qu'on vient de faire
        os.remove(file_name)
