
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

'''
Parametres
'''
heightf, widthf = 640, 1080
seuil = 0.4



#Paramètres
fps = 10
size = (120,90) #Sens inverse au nom du modèle
nb_classes = 10
valeur_slider = 0
#model_name = 'model_convLSTM2D_8_10_75_100_10_2_50_50_1mili.h5'
#path = 'Modele_acc77_bon.h5'
path = 'Saved_model\\model_12_90_120_acc83.h5'


#'old_goods\\model_good_convLSTM2D_10_75_100_10_2_10_50_1mili.h5'
#Broken: [nan]
# fps = 8
# size = (40,30) #Sens inverse au nom du modèle
# nb_classes = 10
# epochs = 1
# batch_size = 20
# pack_size = 50
# learning_rate = 0.01





#-------------------------------------------------------------------------------
'''
Fonctions
'''


'''
Thread
'''
def predict(model):
    '''
    Prédit un résultat de la queue et le remet dans la queue
    '''
    while True:
        #with tf.device('cpu:0'):
        X = q_to_pred.get()
        start_pred = time.time()
        pred = model.predict(X)
        end_pred = time.time()
        print(f"Une prédiction prend {end_pred - start_pred} secondes")
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

         # Button that lets the user take a snapshot
        #self.btn_snapshot=tkinter.Button(window, text="Snapshot", command=self.snapshot).place(x=widthf/3-35, y=heightf-100,width=120,height = 30)
        #self.btn_snapshot2=tkinter.Button(window, text="Snapshot", command=self.snapshot).place(x=2*widthf/3-35, y=heightf-100,width=120,height = 30)
        #self.text1 = tkinter.Label(window, text="inserer ici ce que le modèle a reconnu").place(x=2*widthf/3-35, y=heightf-100)
        #self.btn_snapshot.pack(anchor=tkinter.CENTER, expand=True)
        self.btn_stop_pred = tkinter.Button(window, text="Stop Prediction",
                command=self.stop_pred).place(x=2*widthf/3, y = 80)

        self.btn_new_window = tkinter.Button(window, text = "Ouvrir une nouvelle fenêtre",
                command = self.open_window).place(x = 80, y = 80)


        #Pas besoin de pack parce qu'on le place direct
        # #liste de tous les gestes sur la droite de l'écran
        #
        # label = pd.read_csv("DATA\\labels_uses.csv")
        #
        # for i,labels in enumerate(label.values):
        #     self.text1 = tkinter.Label(window, text=labels[0]).place(x=2*widthf/3, y=(30*i)+100)
        #
        self.movs = []
        self.stop_showing = False
        self.is_freezed = False
        self.example_vid = None #Elle n'existe pas au début, on ne la fabrique que si c'est nécessaire


        t_show = threading.Thread(target=self.get_prediction,daemon=True)
        t_show.start()

         # After it is called once, the update method will be automatically called every delay milliseconds
         #Affichage
        self.delay = 15
        self.update()

        #Mouvement
        self.delay_fps = int(1000/fps)
        self.update_mov()





        self.window.mainloop()

    def stop_pred(self):
        self.stop_showing = not self.stop_showing
        if self.stop_showing:
            self.text1 = tkinter.Label(self.window, text="Affichage arrêté").place(x=2*widthf/3+150, y = 80)
        else:
            self.text1 = tkinter.Label(self.window, text=" "*50).place(x=2*widthf/3+150, y = 80)

    def update(self):
         # Get a frame from the video source
         if not self.is_freezed:
             ret, frame = self.vid.get_frame()

             if ret:
                 #Affichage de l'image
                 self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
                 self.canvas.create_image(0, 0, image = self.photo, anchor = tkinter.NW)


         self.window.after(self.delay, self.update)
    #
    # def update_new_window(self):
    #      # Get a frame from the video source
    #     new_window_set = 0
    #     if new_window_set == 1:
    #
    #         for i in len(self.imgs) :
    #          #Affichage de l'image
    #             self.photo_vid = PIL.ImageTk.PhotoImage(image = self.imgs[i])
    #             self.canvas.create_image(0, 0, image = self.photo_vid, anchor = tkinter.NW)
    #
    #     self.window.after(self.delay, self.update)


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
                    print("Le fichier n'a pas pu être ouvert")
            except:
                print(f"IL Y A UN PROBLEME AVEC LA VIDEO: {video_name}")
            cap.release()
            self.example_vid.append(imgs)
        #print("J'ai réussi à importer toutes les vidéos") #DEBUG

    def open_window(self):
        self.freeze()
        new_window = tkinter.Toplevel(self.window)
        new_window.grab_set() #Force le focus sur cette fenetre
        new_window.geometry("500x500")
        new_window.title("New Window")
        lbl = tkinter.Label(new_window, text="Je suis une nouvelle fenetre")
        lbl.pack()


        #Liste des vidéos
        self.canvas_ex = tkinter.Canvas(new_window,width =320,height = 240)
        self.canvas_ex.pack()
        self.create_example_videos(20) # 20 fps


        def show_videos():
            if self.example_vid == None:
                print("Pas de videos")
                return
            vid_len = len(self.example_vid[0])
            tot_len = vid_len * len(self.example_vid)
            if self.count_img_example >= tot_len:
                self.count_img_example = 0



            img = self.example_vid[int(self.count_img_example / vid_len)][int(self.count_img_example % vid_len)]
            #img = self.example_vid[0][5]
            self.photo_vid = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(img))
            self.canvas_ex.create_image(0, 0, image = self.photo_vid, anchor = tkinter.NW)

            self.count_img_example += 1
            new_window.after(50, show_videos)

        self.count_img_example = 0
        show_videos()

        # Ajouter un nouveau label
        def new_label():
            #Idéalement, il faudrait un peu de vérification du nom inséré
            #(retirer les virgules par exemple, mais ça n'est pas le but du projet)
            csv = "DATA\\labels_uses.csv"
            lbl_name = entry_val.get()
            labels_list = pd.read_csv(csv)
            if lbl_name in labels_list:
                #Il y a un problème, il ne passe pas ici et ajoute deux fois le nom
                text_confirm_new_label = tkinter.Label(new_window, text = "Ce label existe déjà")
                return

            labels_csv = open(csv,"a")
            labels_csv.write("\n" + lbl_name) #Ajoute une nouvelle info sur une nouvelle ligne
            labels_csv.close()
            text_confirm_new_label = tkinter.Label(new_window, text = "Label ajouté")
            text_confirm_new_label.pack()


        lbl_new_vid = tkinter.Label(new_window, text = "Ajouter un nouveau label")
        lbl_new_vid.pack() #DEBUG A PLACER PAR NICO

        entry_val = tkinter.StringVar() #Variable spéciale nécessaire pour l'entry_val
        in_new_vid = tkinter.Entry(new_window, textvariable = entry_val )
        in_new_vid.pack()
        btn_new_vid =tkinter.Button(new_window, text = "Confirmer le nouveau label",
                                    command = new_label)
        btn_new_vid.pack()


        #Filmer des vidéos





        # Fermeture de la fenêtre
        def exit_window():
            new_window.destroy()
            new_window.update()
            self.text_freezed_window.destroy() #Supprime le message qui était sur l'autre fenêtre
            self.is_freezed = False
        # Lorsqu'on quitte sur la croix, ça exécute la fonction
        new_window.protocol("WM_DELETE_WINDOW", exit_window)

    def freeze(self):
        '''
        Affiche un message sur la fenêtre principale pour bloquer
        '''
        self.is_freezed = True
        self.text_freezed_window = tkinter.Label(self.window, text="FREEZE, fermez l'autre \n fenetre pour defreeze")
        self.text_freezed_window.pack()




    def update_mov(self):
        if not self.is_freezed:
            ret, prev = self.vid.get_frame()
            time.sleep(0.01) #Methode de bourrin, il faudrait autre chose
            ret, frame = self.vid.get_frame()
            gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            prev_gray = cv2.cvtColor(prev,cv2.COLOR_BGR2GRAY)
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
            #normalized = cv2.normalize(resized,0,1,cv2.NORM_MINMAX)
            self.movs.append(normalized)

        #Je n'ai pas besoin de mettre ce qui suit dans le if mais j'amagine que
        # c'est plus rapide de ne pas passer dessus (sans arguments ni preuves)

            #Lance la prédiction si on a le bon nombre d'images
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
            self.text1 = tkinter.Label(self.window, text=str(len(self.movs))+" frames already captured      ").place(x=2*widthf/3, y=330)

        self.window.after(self.delay_fps,self.update_mov)

    def get_prediction(self):
        '''
        Affiche un résultat
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
            print("La première valeur est ", sortable_pred[0])
            if not self.stop_showing:
                if sortable_pred[0] >=(seuil,):
                    for i,(val, index) in enumerate(sortable_pred):
                        #pourcent = f"{val:4.3f}"
                        nom = labels_n[index]
                        texte = str(val) +"  :  " + nom + " "*70
                        self.text1 = tkinter.Label(self.window, text=texte).place(x=2*widthf/3, y=(30*i)+100)
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



#Fabrication du dictionnaire de noms
labels_name = pd.read_csv("DATA\\labels_uses.csv")
labels_n = {}
for i,label in enumerate(labels_name.values):
    labels_n[label[0]] = i
    labels_n[i] = label[0]



#queue
q_to_pred = Queue() #X vers la prédiction
q_pred = Queue() #pred vers l'affichage

#Thread
t_pred = threading.Thread(target=predict,args=((model,)),daemon= True)
t_pred.start()


#Quelques variables globales utiles
n_frames = int(fps*2.4)


 # Create a window and pass it to the Application object
App(tkinter.Tk(), "Tkinter and OpenCV")

# valeur_slider = slider.get()
# slider = tkinter.Scale(self.window, from_=0, to=100, tickinterval = 20,orient="horizontal",label = "Thershold value",length = 300)
# slider.place(x=300,y=550)
