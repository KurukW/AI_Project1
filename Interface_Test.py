import tkinter as tk
import cv2
import PIL.Image, PIL.ImageTk
import time
import pandas as pd
from tkinter import ttk
from tkinter.messagebox import showerror


heightf, widthf = 600, 800

class Fenetre_video:
    def __init__(self, window, window_title, video_source=0):
        self.window = window
        self.window.title(window_title)
        self.window.geometry(str(widthf) +"x"+str(heightf))
        self.video_source = video_source

         # open video source (by default this will try to open the computer webcam)
        self.vid = MyVideoCapture(self.video_source)

        button = tk.Button(window, text="new window", command=self.open_window)
        button.pack()

        #liste de tous les gestes sur la droite de l'écran

        label = pd.read_csv("DATA\\labels_uses.csv")

        for i,labels in enumerate(label.values):
            self.text1 = tk.Label(window, text=labels[0]).place(x=2*widthf/3, y=(30*i)+100)

         # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay = 15
        self.update()

        self.window.mainloop()

    def update(self):
         # Get a frame from the video source
         ret, frame = self.vid.get_frame()

         if ret:
             self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))

         self.window.after(self.delay, self.update)

    def open_window(self):
        self.freeze()
        new_window = tk.Toplevel(self.window)
        new_window.grab_set() #Force le focus sur cette fenetre
        new_window.geometry("250x250")
        new_window.title("New Window")
        lbl = tk.Label(new_window, text="Je suis une nouvelle fenetre")
        lbl.pack()

        def exit_window():
            new_window.destroy()
            new_window.update()

        #Lorsqu'on quitte sur la croix, ça exécute la fonction
        new_window.protocol("WM_DELETE_WINDOW", exit_window)

    def freeze(self):
        self.text2 = tk.Label(self.window, text="FREEZE, fermez l'autre \n fenetre pour defreeze",
        font=("Helvetica", 30)).place(heightf/2, widthf/2)




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



if __name__ == "__main__":
    app = Fenetre_video(tk.Tk(),"fenetre Principale")
    app.mainloop()
