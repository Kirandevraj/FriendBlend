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
    face_cascade = cv2.CascadeClassifier('lib/haarcascade_frontalface_default.xml')
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
        x1 = int(x-int(1.5*w))
        if x1<0:
            x1=0
        y1 = int(y-h)
        if y1<0:
            y1=0
        x2 = x+int(2.2*w)
        if x2>img.shape[1]:
            x2=img.shape[1]
        y2 = img.shape[0]
        area = ((x2-x1) * (y2-y1))
        if dist < area:
            dist = area
            list1 = [(x1,y1,x2,y2,w,h)]
    img = cv2.rectangle(img,(list1[0][0],list1[0][1]),(list1[0][2],list1[0][3]),(255,0,0),2)
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    return list1,img


def valid_keypoints(body1,body2,keypoints):
	op_keypoints = keypoints.copy()
	top_left_x1,top_left_y1,bot_right_x1,bot_right_y1,_,_=body1[0]
	top_left_x2,top_left_y2,bot_right_x2,bot_right_y2,_,_=body2[0]
	for i in range(len(keypoints)):
		point = keypoints[i].pt
		if (((point[0]<top_left_x1 or point[0]>bot_right_x1) or (point[1]<top_left_y1 or point[1]>bot_right_y1)) and ((point[0]<top_left_x2 or point[0]>bot_right_x2) or (point[1]<top_left_y2 or point[1]>bot_right_y2))):
			continue
		else:
			op_keypoints.remove(keypoints[i])
	return op_keypoints
