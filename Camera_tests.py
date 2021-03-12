import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import cv2


#face_cascade = cv2.CascadeClassifier('C:\\Users\\william\\Documents\\Scolaire\\M1\\Systèmes_Intelligents\\03_Image and video Processing\\Images and videos\\DATA\\haarcascades\\haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier('DATA\\haarcascade_frontalface_default.xml')
hand_cascade = cv2.CascadeClassifier('DATA\\hand_a.xml') #Pas très efficace


# ----------------------------------------------------------------------------
'''FONCTIONS '''
def detect_face(img): #Detect and draw on the face

    face_img = img.copy()
    face_rects = face_cascade.detectMultiScale(face_img)
    #CETTE LIGNE POSE PROBLEME

    #Tant qu'il y a des paquest de 4 valeurs, on dessine le rectangle
    #Chaque visage est défini par un tuple de 4 valeurs
    for (x,y,w,h) in face_rects:
        cv2.rectangle(face_img, (x,y), (x+w,y+h), (255,255,255), 10)

    return face_img


def detect_hand(img):

    hand_img = img.copy()
    hand_rects = hand_cascade.detectMultiScale(hand_img)

    for (x,y,w,h) in hand_rects:
        cv2.rectangle(hand_img, (x,y), (x+w,y+h), (255,0,0), 10)
        # cv2.circle(hand_img,(x,y),int((w+h)/2),(255,0,0),5)
    return hand_img


def detection(img):
    img = detect_face(img)
    img = detect_hand(img)

    return img


## ----------------------------------------------------------------------------
''' MAIN CODE '''

# Connects to your computer's default camera
cap = cv2.VideoCapture(0)


# Automatically grab width and height from video feed
# (returns float which we need to convert to integer for later on!)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

while True:

    # Capture frame-by-frame
    ret, frame = cap.read()

    # # Our operations on the frame come here
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #
    # # Display the resulting frame
    # cv2.imshow('frame',gray)

    result = detection(frame)

    # Display the resulting frame
    cv2.imshow('frame',result)

    # This command let's us quit with the "q" button on a keyboard.
    # Simply pressing X on the window won't work!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture and destroy the windows
cap.release()
cv2.destroyAllWindows()
