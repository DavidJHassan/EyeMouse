import cv2
import numpy as np
from pymouse import PyMouse

m = PyMouse()
vid = cv2.VideoCapture(0)

screen_size = m.screen_size()
pos = m.position()
mx = pos[0]
my = pos[1]

while True:
    flag, frame = vid.read()
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
    
    if flag:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        eyes = None
        center = []
        xpt = []
        nxpt = []
        ypt =[]
        nypt = []
        mask = np.array([])
        # Find face
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x,y), (x+w,y+h/2), (255,0,0), 2)
            roi_gray = np.zeros(gray.shape, np.uint8)
            roi_color = np.zeros(gray.shape, np.uint8)
            mask = np.zeros(gray.shape, np.uint8)
            roi_gray[y:y+h/2, x:x+w] = gray[y:y+h/2, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)

            # Find one eye. We want to do calculations using one eye for simplicity
            for (ex, ey, ew, eh) in eyes:
                mask[ey:ey+eh, ex:ex+ew] = roi_gray[ey:ey+eh, ex:ex+ew]
                center.append(mask[ey+eh/2, ex+ew/2])
                # Grab some points around the eyes
                for i in range(1, 10):
                    xpt.append(mask[ey+eh/2, ex+ew/2 + i])
                    nxpt.append(mask[ey+eh/2, ex+ew/2 - i])
                    ypt.append(mask[ey+eh/2 + i, ex+ew/2])
                    nypt.append(mask[ey+eh/2 - i, ex+ew/2])
                break

            # Average the points around the eyes to try and
            # determine the eyes looking direction
            avgx = np.mean(xpt) 
            avgnx = np.mean(nxpt) 
            avgy = np.mean(ypt) - 5
            avgny = np.mean(nypt)

            print center, avgx, avgnx

            # Thresholding to try and remove excess pixels that are not needed
            mask[mask < 100] = 255
            mask[mask > 125]  = 255
            mask[mask != 255] = 0

            # Try and make sure that the averages are different
            # enough to be signicant based on some arbitary value 10
            if (avgx - avgnx) < -10:
                mx -= 5
            elif (avgx - avgnx) > 10:
                mx += 5

            if (avgy - avgny) < -10:
                my += 5
            elif (avgy - avgny) > 10:
                my -= 5

            # Update mouse
            m.move(mx, my)
            
    if cv2.waitKey(5) == 27:
        break


cv2.destroyAllWindows()
