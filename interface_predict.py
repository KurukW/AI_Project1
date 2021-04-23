import tkinter
import cv2
import PIL.Image, PIL.ImageTk
import time
import pandas as pd
import sys
import numpy as np
import threading
from queue import Queue
from tensorflow import keras
import tensorflow as tf
from DATA.data_fab import * #Afin d'ajouter une nouvelle video

'''
Parametres
'''
heightf, widthf = 640, 1080
seuil = 0.4 #Affichage des prédictions avec un résultat supérieur au seuil

#Paramètres
fps = 10 #Ne plus modifier, c'est définitif maintenant
size = (120,90) #Sens inverse au nom du modèle
nb_classes = 10
path = 'Saved_model\\model_12_90_120_acc83.h5' #Model final

#-------------------------------------------------------------------------------

'''
Thread
'''
def predict(model):
    '''
    Prédit le mouvement avec les images de "q_to_pred"
    et met le résultat dans "q_pred" qui est affiché par la méthode
    "q_pred" dans la class APP
    '''
    while True:
        #with tf.device('cpu:0'):
        X = q_to_pred.get()
        start_pred = time.time()
        pred = model.predict(X)
        end_pred = time.time()
        #print(f"Une prédiction prend {end_pred - start_pred} secondes") #DEBUG
        q_pred.put(pred)


#-------------------------------------------------------------------------------
'''
Class
'''

class App:
    def __init__(self, window, window_title, video_source=0):
        self.window = window
        self.window.title(window_title)
        self.window.geometry(str(widthf) +"x"+str(heightf))
        self.video_source = video_source

         # open video source (by default this will try to open the computer webcam)
        self.vid = MyVideoCapture(self.video_source)

         # Create a canvas that can fit the above video source size
        self.canvas = tkinter.Canvas(window, width = self.vid.width, height = self.vid.height)
        self.canvas.pack()
        self.canvas.place(x = 50,y = 0)

        self.btn_stop_pred = tkinter.Button(window, text="Stop Prediction",
                command=self.stop_pred).place(x=2*widthf/3, y = 60)

        self.btn_new_window = tkinter.Button(window, text = "Configuration",
                command = self.open_window).place(x=2*widthf/3, y = 20)

        #Quelques variables
        self.movs = [] #Images à prédire
        self.stop_showing = False #Fige l'affichage des prédictions
        self.is_freezed = False #Fige toute la fenêtre
        self.example_vid = None #Elle n'existe pas au début, on ne la fabrique que si c'est nécessaire


        #Thread de l'affichage
        t_show = threading.Thread(target=self.get_prediction,daemon=True)
        t_show.start()

         # After it is called once, the update method will be automatically called every delay milliseconds
         #Affichage
        self.this_frame = False #Je prend une image sur deux pour la prédiction
        self.delay = 50 #20 fps
        self.update()

        self.window.mainloop()


    def stop_pred(self):
        '''
        freeze l'affichage des prédictions
        '''
        self.stop_showing = not self.stop_showing
        if self.stop_showing:
            self.text_display = tkinter.Label(self.window, text="Stopped display").place(x=2*widthf/3+150, y = 80)
        else:
            self.text_display = tkinter.Label(self.window, text=" "*50).place(x=2*widthf/3+150, y = 80)

    def update(self):
        '''
        Méthode appelée toutes les 15 ms : permet l'affichage de la caméra ainsi que le traitement des images
        '''
         # Get a frame from the video source
        if not self.is_freezed:
             ret, frame = self.vid.get_frame()

             if ret:
                 #Affichage de l'image
                 self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
                 self.canvas.create_image(0, 0, image = self.photo, anchor = tkinter.NW)

                 #Mouvement
                 if self.this_frame:
                    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                    prev_gray = cv2.cvtColor(self.prev,cv2.COLOR_BGR2GRAY)
                    #Détection de mouvement
                    diff = cv2.absdiff(gray,prev_gray)
                    #Réduire la taille
                    resized = cv2.resize(diff, dsize=size, interpolation=cv2.INTER_LINEAR)
                    #Normaliser
                    res_max = resized.max()
                    if res_max != 0:
                        normalized = resized/float(resized.max())
                    else:
                        normalized = resized
                    self.movs.append(normalized)

                #Envoie à la prédiction
                 if len(self.movs) > (n_frames):
                     X = self.movs[:n_frames]
                     del self.movs[:n_frames]

                     #Traitement de X
                     X = np.array(X)
                     X = np.expand_dims(X, axis=len(X.shape)) #Ajoute un channel
                     X = np.expand_dims(X, axis = 0)

                     #Predict de X
                     q_to_pred.put(X)

                 # afficher le nombre de frames
                 self.text_n_frames = tkinter.Label(self.window, text=str(len(self.movs))+" frames already captured      ").place(x=2*widthf/3, y=330)


                 self.this_frame = not self.this_frame #J'inverse pour prendre une image sur deux
                 self.prev = frame

        #Appel à nouveau cette methode dans "delay" millisecondes
        self.window.after(self.delay, self.update)

    def create_example_videos(self, fps = -1):
        '''
        Créer la liste d'images qui vont défiler en boucle dans la deuxième fenêtre
        '''
        if not self.example_vid == None:
            return

        self.example_vid = []

        csv_ref = pd.read_csv("DATA\\labels_example.csv")
        img = None
        for label, video_name in csv_ref.values:
            cap = cv2.VideoCapture("DATA\\Video_example\\" + video_name)
            #print("la video est ", video_name) #DEBUG
            try:
                fps_actu = cap.get(cv2.CAP_PROP_FPS)
                if fps <= -1: fps = fps_actu #Je peux ne pas donner de fps et ça va prendre le nombre d'fps initial
                ecart_voulu = int(1000/fps)
                ecart_initial = int(1000/fps_actu)
                imgs = []

                while(cap.isOpened()):
                    ret, frame = cap.read()

                    if ret: #Sinon ça plante quand il n'y a plus d'images
                        #Récupère seulement certaines images
                        t_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                        modulo = t_ms % ecart_voulu
                        if modulo < ecart_initial:
                            #Isoler les images
                            colored =  cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            resized = cv2.resize(colored, dsize=(320,240), interpolation=cv2.INTER_LINEAR)
                            imgs.append(resized)
                    else: #Va jusqu'au bout de la vidéo
                        break
                else:
                    print("Le fichier n'a pas pu être ouvert") #Error
            except:
                print(f"IL Y A UN PROBLEME AVEC LA VIDEO: {video_name}") #Error
            cap.release()
            self.example_vid.append((imgs,label))
        #print("J'ai réussi à importer toutes les vidéos") #DEBUG

    def open_window(self):
        '''
        méthode générale pour la deuxième fenetre
        '''
        #Freeze la première fenêtre afin d'économiser la cpu et augmenter les performances
        self.freeze()
        new_window = tkinter.Toplevel(self.window)
        new_window.grab_set() #Force le focus sur cette fenetre
        new_window.geometry("500x500")
        new_window.resizable(False,False) #Fixe la taille de la fenêtre
        new_window.title("Configuration")
        #Label de la video qui passe actuellement
        self.label_video_name = tkinter.Label(new_window,text = 'loading the video', font = ("Helvetica", 18))
        self.label_video_name.pack()

        #Liste des vidéos
        self.canvas_ex = tkinter.Canvas(new_window,width =320,height = 240)
        self.canvas_ex.pack()
        self.create_example_videos(20) # 20 fps

        def add_vid():
            '''
            methode pour ajouter une video
            '''
            del self.vid
            save_video('DATA\\Videos', 'DATA\\labels.csv', 'DATA\\labels_uses.csv',25,2)
            self.vid = MyVideoCapture(self.video_source)
        btn_add_vid = tkinter.Button(new_window, text = "Add a video to the training set",
                                command = add_vid)
        btn_add_vid.place(x = 30, y = 400)


        def show_videos():
            '''
            défilement automatique des videos sur la seconde fenetre
            '''
            if self.example_vid == None:
                print("Pas de videos")
                return
            vid_len = len(self.example_vid[0][0])
            tot_len = vid_len * len(self.example_vid)
            if self.count_img_example >= tot_len:
                self.count_img_example = 0


            img = self.example_vid[int(self.count_img_example / vid_len)][0][int(self.count_img_example % vid_len)]
            #img = self.example_vid[0][5]
            self.photo_vid = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(img))
            self.canvas_ex.create_image(0, 0, image = self.photo_vid, anchor = tkinter.NW)
            label = self.example_vid[int(self.count_img_example / vid_len)][1]
            self.label_video_name.config(text = label)

            self.count_img_example += 1
            new_window.after(50, show_videos)

        self.count_img_example = 0
        show_videos()

        # Ajouter un nouveau label
        def new_label():
            '''
            ajout d'un nouveau geste
            '''
            #Idéalement, il faudrait un peu de vérification du nom inséré
            #(retirer les virgules par exemple, mais ça n'est pas le but du projet)
            csv = "DATA\\labels_uses.csv"

            lbl_name = entry_val.get()
            labels_list = pd.read_csv(csv)
            if lbl_name in labels_list.values:
                #Il y a un problème, il ne passe pas ici et ajoute deux fois le nom
                text_confirm_new_label.config(text ="This label already exists")
                return
            labels_csv = open(csv,"a")
            labels_csv.write("\n" + lbl_name) #Ajoute une nouvelle info sur une nouvelle ligne
            labels_csv.close()
            text_confirm_new_label.config(text ="Label added")


        #Confirmation d'ajout d'un label
        text_confirm_new_label = tkinter.Label(new_window)
        text_confirm_new_label.pack()


        lbl_new_vid = tkinter.Label(new_window, text = "Add a new label")
        lbl_new_vid.pack() #DEBUG A PLACER PAR NICO

        entry_val = tkinter.StringVar() #Variable spéciale nécessaire pour l'entry_val
        in_new_vid = tkinter.Entry(new_window, textvariable = entry_val )
        in_new_vid.pack()
        btn_new_vid =tkinter.Button(new_window, text = "Comfirm the new label",
                                    command = new_label)
        btn_new_vid.pack()


        def train():
            '''
            methode pour l'entrainement du modèle, le code n'est pas implémenter car l'entrainement est trop long ( min 10 heures)
            '''
            print("Model training ...")
            #On ne peut pas train le modele, ça prend 10h
            time.sleep(2)
            print("Model trained")
        #Bouton pour retrain le model
        btn_train = tkinter.Button(new_window, text = "Retrain the model",
                                    command = train)
        btn_train.place(x = 350, y = 400)

        # Fermeture de la fenêtre
        def exit_window():
            new_window.destroy()
            new_window.update()
            self.text_freezed_window.destroy() #Supprime le message qui était sur l'autre fenêtre
            self.is_freezed = False #Defreeze la fenêtre principale

        # Lorsqu'on click sur la croix, ça exécute la fonction
        new_window.protocol("WM_DELETE_WINDOW", exit_window)

    def freeze(self):
        '''
        Affiche un message sur la fenêtre principale pour bloquer
        '''
        self.is_freezed = True
        self.text_freezed_window = tkinter.Label(self.window, text="FREEZE, close the other \n window to defreeze",
                                    font = ('Helvetica', 30), bg = 'red')
        self.text_freezed_window.pack()


    def get_prediction(self):
        '''
        Affiche un résultat que la fonction "predict" a calculé
        '''
        while True:

            pred = q_pred.get()
                # sortable_pred = []
                #Classement dans l'ordre des prédictions
                # for i,elt in enumerate(pred):
                #     sortable_pred.append((elt,i)) #Je met la valeur en premier pour trier plus facilement
            # En une ligne:
            sortable_pred = [(elt,i) for i,elt in enumerate(pred[0])]
            sortable_pred.sort(reverse = True)

            #Affichage des résultat
            #print("La première valeur est ", sortable_pred[0]) #Debug
            if not self.stop_showing:
                if sortable_pred[0][0] >= seuil:
                    for i,(val, index) in enumerate(sortable_pred):
                        #pourcent = f"{val:4.3f}"
                        nom = labels_n[index]
                        texte = str(round((val*100),2)) + " %" +"  :  " + nom + " "*70
                        self.text1 = tkinter.Label(self.window, text=texte).place(x=2*widthf/3, y=(30*i)+100)

                        if i == 0:
                            self.text_big = tkinter.Label(self.window, text=nom + " "*70 ,font = ('Helvetica', 30)).place(x=50, y=550)
                        if i ==2:
                            break
                else:
                    self.text1 = tkinter.Label(self.window, text=" "*107).place(x=2*widthf/3, y=100)
                    self.text1 = tkinter.Label(self.window, text="The prediction is not high enough"+" "*50).place(x=2*widthf/3, y=130)
                    self.text1 = tkinter.Label(self.window, text=" "*107).place(x=2*widthf/3, y=160)


class MyVideoCapture:
    def __init__(self, video_source=0):
         # Open the video source
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)

         # Get video source width and height
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)

    def get_frame(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                 # Return a boolean success flag and the current frame converted to BGR
                return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                return (ret, None)
        else:
            return (ret, None)

     # Release the video source when the object is destroyed
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()
#-------------------------------------------------------------------------------
'''
Main
'''
#Import du modèle

try:
    model = keras.models.load_model(path)
    #keras.models.load_model('Saved_model\\modele_stolen_compile')
    print("modele importé avec succès")
except:
    print("Erreur lors du chargement du modele",
    "vérifiez que les paramètres sont bons et que le modele existe")
    print("Je tentais d'importer le modele:",path)
    sys.exit()


#Fabrication du dictionnaire de noms. Il sert à l'affichage des prédictions
labels_name = pd.read_csv("DATA\\labels_uses.csv")
labels_n = {}
for i,label in enumerate(labels_name.values):
    labels_n[label[0]] = i
    labels_n[i] = label[0]


#queue
q_to_pred = Queue() #X vers la prédiction
q_pred = Queue() #pred vers l'affichage

#Thread de la prediction
t_pred = threading.Thread(target=predict,args=((model,)),daemon= True)
t_pred.start()


#Quelques variables globales utiles
n_frames = int(fps*2.4)


 # Create a window and pass it to the Application object
App(tkinter.Tk(), "Tkinter and OpenCV")
