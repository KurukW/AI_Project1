
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


#Paramètres
fps = 10
size = (100,75) #Sens inverse au nom du modèle
nb_classes = 10
#model_name = 'model_convLSTM2D_8_10_75_100_10_2_50_50_1mili.h5'
path = 'Modele_acc77_bon.h5'

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
        self.btn_stop_pred = tkinter.Button(window, text="Freeze", command=self.stop_pred).place(x=2*widthf/3, y = 80)
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

    def snapshot(self):
         # Get a frame from the video source
         ret, frame = self.vid.get_frame()

         if ret:
            cv2.imwrite("frame-" + time.strftime("%d-%m-%Y-%H-%M-%S") + ".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    def stop_pred(self):
        self.stop_showing = not self.stop_showing
        if self.stop_showing:
            self.text1 = tkinter.Label(self.window, text="I'm freezed").place(x=2*widthf/3+150, y = 80)
        else:
            self.text1 = tkinter.Label(self.window, text=" "*50).place(x=2*widthf/3+150, y = 80)

    def update(self):
         # Get a frame from the video source
         ret, frame = self.vid.get_frame()


         if ret:
             #Affichage de l'image
             self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
             self.canvas.create_image(0, 0, image = self.photo, anchor = tkinter.NW)

         self.window.after(self.delay, self.update)

    def update_mov(self):
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
            normalized = resized.copy()
        #normalized = cv2.normalize(resized,0,1,cv2.NORM_MINMAX)
        self.movs.append(normalized)


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
            for i,(val, index) in enumerate(sortable_pred):
                #pourcent = f"{val:4.3f}"
                nom = labels_n[index]
                texte = str(val) +"  :  " + nom + " "*70
                if not self.stop_showing:
                    self.text1 = tkinter.Label(self.window, text=texte).place(x=2*widthf/3, y=(30*i)+100)


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
