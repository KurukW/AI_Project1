'''
stolen
https://solarianprogrammer.com/2018/04/21/python-opencv-show-video-tkinter-window/


Intègre une vidéo opencv dans tkinter


'''
import tkinter
import cv2
import PIL.Image, PIL.ImageTk
import time

ratio = 1

class App:
 def __init__(self, window, window_title, video_source=0):
     self.window = window
     self.window.title(window_title)
     self.video_source = video_source

     # open video source (by default this will try to open the computer webcam)
     self.vid = MyVideoCapture(self.video_source)

     # Create a canvas that can fit the above video source size
     self.canvas = tkinter.Canvas(window, width = int(self.vid.width*ratio), height = int(self.vid.height*ratio))
     self.canvas.pack()

     self.canvas2 = tkinter.Canvas(window, width = int(self.vid.width*ratio), height = int(self.vid.height*ratio))
     self.canvas2.pack()

     # Button that lets the user take a snapshot
     self.btn_snapshot=tkinter.Button(window, text="Snapshot", width=50, command=self.snapshot)
     self.btn_snapshot.pack(anchor=tkinter.CENTER, expand=True)

     # After it is called once, the update method will be automatically called every delay milliseconds
     self.delay = 15 #Délai entre chaque image
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
     _, movement = self.vid.get_movement()

     if ret:
         self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
         self.photo2 = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(movement))
         self.canvas.create_image(0, 0, image = self.photo, anchor = tkinter.NW)
         self.canvas2.create_image(0, 0, image = self.photo2, anchor = tkinter.NW)

     self.window.after(self.delay, self.update)
     #après x ms, on appelle la fonction update. Donc cette fonction s'appelle tt le temps


class MyVideoCapture:
 def __init__(self, video_source=0):
     # Open the video source
     self.vid = cv2.VideoCapture(video_source)
     if not self.vid.isOpened():
         raise ValueError("Unable to open video source", video_source)

     # Get video source width and height
     self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
     self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)

     # Mes ajouts
     _,self.frame = self.vid.read()
     #Je le fais une première fois pour calculer mon mouvement dès le début
     #J'ai aussi rendu frame comme variable de mon objet afin de pouvoir
     # la comparer dans plusieurs fonctions différentes


     self.thresh = 10 # valeur de threshold pour le mouvement
     #La valeur change le bruit mais aussi la qualité de la détection


 def get_frame(self):
     if self.vid.isOpened():
         ret, self.frame = self.vid.read()
         if ret:
             # Return a boolean success flag and the current frame converted to BGR
             colored = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
             resized = small = cv2.resize(colored,
                                 dsize=(int(self.width*ratio),int(self.height*ratio)),
                                 interpolation=cv2.INTER_CUBIC)
             return (ret, resized)
         else:
             return (ret, None)
     else:
         return (ret, None)

 def get_movement(self):
     '''
     Ma fonction
     Donne une image du mouvement uniquement, fonction développée dans "old_Skeletonization\\movement détection"
     '''
     if self.vid.isOpened():
        prev = self.frame.copy()
        ret, self.frame = self.vid.read()
        if ret:
            # Return a boolean success flag and the current frame converted to BGR
            diff = cv2.absdiff(self.frame,prev)
            _,diff_thresh = cv2.threshold(diff,self.thresh,255,cv2.THRESH_BINARY)
            movement = cv2.cvtColor(diff_thresh,cv2.COLOR_BGR2GRAY)
            frame_rgb = cv2.cvtColor(self.frame,cv2.COLOR_BGR2RGB)
            movement_colored = cv2.bitwise_and(frame_rgb,frame_rgb,mask = movement)
            #resize
            small = cv2.resize(movement_colored,
                                dsize=(int(self.width*ratio),int(self.height*ratio)),
                                interpolation=cv2.INTER_CUBIC)
            return (ret, small)
        else:
            return (ret, None)
     else:
        return (ret, None)


 # Release the video source when the object is destroyed
 def __del__(self):
     if self.vid.isOpened():
         self.vid.release()

if __name__ == '__main__':
    # Create a window and pass it to the Application object
    App(tkinter.Tk(), "Tkinter and OpenCV")
