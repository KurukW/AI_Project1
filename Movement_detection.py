import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import cv2
from random import randint
from skimage.measure import compare_ssim
# ----------------------------------------------------------------------------
'''FONCTIONS '''
def applyKernel(img):
    kernel = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]) #Edge detection
    return cv2.filter2D(img,-1,kernel)

def diff(now,prev,threshold = 150):
    '''
    Now et Prev sont deux images qui se suivent.
    Elles doivent être en nuance de gris
    '''
    (score,diff) = compare_ssim(now,prev,full=True)
    next = (diff*255).astype("uint8") #Utile pour le threshold
    result = cv2.threshold(dif,threshold,255,cv2.THRESH_BINARY_INV)
    return result[1]


def oneColor(img,n):
    new_img = img.copy()
    for i in range(0,3):
        if i == n:
            continue
        new_img[:,:,i] = 0
    return new_img

## ----------------------------------------------------------------------------
''' MAIN CODE '''

# Connects to your computer's default camera
cap = cv2.VideoCapture(0)


# Automatically grab width and height from video feed
# (returns float which we need to convert to integer for later on!)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
ret, frame = cap.read()


while True:

    # Capture frame-by-frame
    prev = frame #J'enregistre la dernière image
    ret, frame = cap.read()

    # # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.cvtColor(prev,cv2.COLOR_BGR2GRAY)
    #gray = cv2.cvtColor(diff(frame,prev), cv2.COLOR_BGR2GRAY)

    dif = diff(gray,prev_gray)
    #cv2.imshow('frame',applyKernel(frame))
    #result = cv2.threshold(gray,110,255,cv2.THRESH_BINARY)

    #cv2.imshow('window',result[1])
    cv2.imshow('frame',dif)
    cv2.imshow('window',good_result[1])




    # This command let's us quit with the "q" button on a keyboard.
    # Simply pressing X on the window won't work!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture and destroy the windows
cap.release()
cv2.destroyAllWindows()
