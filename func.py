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


def keypoints_orb_descriptor(img, kp, n=1000):
	orb = cv2.ORB_create(nfeatures=n)
	kp, des = orb.compute(img, kp)
	return kp, des

def keypoint_bf_matcher(des1, des2, n=40):
	bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
	matches = bf.match(des1,des2)
	matches = sorted(matches, key = lambda x:x.distance)
	min_dist = matches[0].distance	
	if (len(matches) < 500):
		n = 20
	return matches[0:n]

def extract_matched_points(dmatches, kpts1, kpts2):
    src_pts  = np.float32([kpts1[m.queryIdx].pt for m in dmatches]).reshape(-1,1,2)
    dst_pts  = np.float32([kpts2[m.trainIdx].pt for m in dmatches]).reshape(-1,1,2)
    return src_pts, dst_pts

def calculate_homography_matrix(pts_src, pts_dst):
    h, status = cv2.findHomography(pts_src, pts_dst)
    return h

def warp_perspective(img_src, h):
    im_out = cv2.warpPerspective(img_src, h, (img_src.shape[1],img_src.shape[0]))
    return im_out

def transform_points(pt1, homography_matrix):
    new_points = cv2.perspectiveTransform(pt1, homography_matrix)
    new_points[new_points <0] = 0
    new_points = new_points.reshape((new_points.shape[0], new_points.shape[2]))
    return new_points
