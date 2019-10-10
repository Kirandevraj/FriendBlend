
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