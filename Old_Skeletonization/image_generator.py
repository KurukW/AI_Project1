import numpy as np
import cv2



def generate_image(shape = (48,48),type = "rect",point1 = (12,12), size = (24,24)):
    #si je ne met que deux chiffres, c'est noir/blanc.
    #Pour les couleurs, je dois ajouter ,3 à ma shape
    img = np.zeros(shape = shape,dtype = np.uint8)

    if type == "rect":
        cv2.rectangle(img,point1,(point1[0] + size[0], point1[1] + size[1]),(255,255,255),thickness = -1)
    if type == "circ":
        cv2.circle(img,point1,size,(255,255,255),thickness = -1)
    return img




if __name__ == '__main__':
    folder = "images\\"
    img = generate_image((48,48), "rect", (12,6), (24,12))
    cv2.imwrite((folder+"rectangle.jpg"),img)
    img = generate_image((48,48), "circ", (24,24),12)
    cv2.imwrite(folder+"circle.jpg",img)
