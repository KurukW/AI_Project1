# from tkinter import *
# import cv2
# from PIL import Image, ImageTk
#
# height, width = 640, 1080
#
#
# fenetre = Tk()
# fenetre.title("Gesture recognition")
# fenetre.geometry(str(width) +"x"+str(height))
#
# def buttonPressed():
#     text2 = Label(fenetre, text="Bouton appuyé")
#     text2.pack()
#
#
#
# text1 = Label(fenetre, text="inserer ici ce que le modèle a reconnu").place(x=30, y=2*height/3)
# buttonExample = Button(fenetre, text="Allez vas-y appuye", width=30,command=buttonPressed).place(x=30, y=height/3)
#
# cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
#
#
# def show_frame():
#     _, frame = cap.read()
#     frame = cv2.flip(frame, 1)
#     cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
#     img = Image.fromarray(cv2image)
#     imgtk = ImageTk.PhotoImage(image=img)
#     fenetre.imgtk = imgtk
#     fenetre.configure(imgtk)
#     fenetre.after(10, show_frame)
#
# show_frame()
# fenetre.mainloop()


import tkinter
import cv2
import PIL.Image, PIL.ImageTk
import time
import pandas as pd

heightf, widthf = 640, 1080

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
        self.btn_snapshot=tkinter.Button(window, text="Snapshot", command=self.snapshot).place(x=widthf/3-35, y=heightf-100,width=120,height = 30)
        #self.btn_snapshot2=tkinter.Button(window, text="Snapshot", command=self.snapshot).place(x=2*widthf/3-35, y=heightf-100,width=120,height = 30)
        #self.text1 = tkinter.Label(window, text="inserer ici ce que le modèle a reconnu").place(x=2*widthf/3-35, y=heightf-100)
        #self.btn_snapshot.pack(anchor=tkinter.CENTER, expand=True)


        #liste de tous les gestes sur la droite de l'écran

        label = pd.read_csv("DATA\\labels_uses.csv")

        for i,labels in enumerate(label.values):
            self.text1 = tkinter.Label(window, text=labels[0]).place(x=2*widthf/3, y=(30*i)+100)




         # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay = 15
        self.update()

        self.window.mainloop()

    def snapshot(self):
         # Get a frame from the video source
         ret, frame = self.vid.get_frame()

         if ret:
            cv2.imwrite("frame-" + time.strftime("%d-%m-%Y-%H-%M-%S") + ".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    def update(self):
         # Get a frame from the video source
         ret, frame = self.vid.get_frame()

         if ret:
             self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
             self.canvas.create_image(0, 0, image = self.photo, anchor = tkinter.NW)

         self.window.after(self.delay, self.update)


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

 # Create a window and pass it to the Application object
App(tkinter.Tk(), "Tkinter and OpenCV")
