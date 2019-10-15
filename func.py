import cv2
import sys
import numpy as np

def lab_contrast(img, grid_size=7):
	img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
	lab_planes = cv2.split(img_lab)
	clh = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(grid_size,grid_size))
	lab_planes[0] = clh.apply(lab_planes[0])
	img_lab = cv2.merge(lab_planes)
	img_bgr = cv2.cvtColor(img_lab, cv2.COLOR_LAB2BGR)
	return img_bgr

def detect_body(img):
    var = 1.5
    iter=0
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, var, 5)
    while(len(faces)==0):
        iter += 1
        var = var - 0.1
        faces = face_cascade.detectMultiScale(gray, var, 5)
        if iter>=5:
            break
    dist=0
    for (x,y,w,h) in faces:
        x1 = int(x-int(w))
        y1 = int(y-h)
        x2 = x+int(w)
        y2 = img.shape[0]
        area = ((x2-x1) * (y2-y1))
        if dist < area:
            dist = area
            list1 = [(x1,y1,x2,y2,w,h)]
    img = cv2.rectangle(img,(list1[0][0],list1[0][1]),(list1[0][2],list1[0][3]),(255,0,0),2)
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    return list1,img
