import cv,cv2
import numpy as np
vid = cv2.VideoCapture(0)


old_eyes = None

while(1):
	flag,frame = vid.read()
	face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
	eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
	if flag:
		
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		faces = face_cascade.detectMultiScale(gray, 1.3, 5)
		eyes = None
		mask = np.array([])
		for (x,y,w,h) in faces:
			#cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
			cv2.rectangle(frame,(x,y),(x+w,y+h/2),(255,0,0),2)
			roi_gray = np.zeros(gray.shape,np.uint8)
			roi_color = np.zeros(gray.shape,np.uint8)
			mask = np.zeros(gray.shape,np.uint8)
			roi_gray[y:y+h/2, x:x+w] = gray[y:y+h/2, x:x+w]
			#roi_color[y:y+h/2, x:x+w] = frame[y:y+h/2, x:x+w]
			eyes = eye_cascade.detectMultiScale(roi_gray)
			
			for (ex,ey,ew,eh) in eyes:
				mask[ey:ey+eh+10, ex:ex+ew+10] = roi_gray[ey:ey+eh+10, ex:ex+ew+10]
				#cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
				cv2.line(mask, (ex+ew/2,ey+5), (ex+ew/2, ey+eh-5), (255,0,0), 2)
				cv2.line(mask, (ex,ey+eh/2), (ex+ew, ey+eh/2), (255,0,0), 2)

		cv2.imshow('img',frame)
		
		
		if len(mask):
			'''
			circles =  cv2.HoughCircles(mask,cv2.cv.CV_HOUGH_GRADIENT, 2, 100, np.array([]), 30, 80,0,140)
			if circles is not None:
					for c in circles[0]:
							cv2.circle(mask, (c[0],c[1]), c[2], (0,255,0),2)
			'''
			cv2.imshow('eyes',mask)
			
			
			'''	
			if old_eyes != None:
				
				cv2.imshow("diff", difference)
		    '''
			old_eyes = mask
	if cv2.waitKey(5)==27:
		break
		
		
cv2.destroyAllWindows()

