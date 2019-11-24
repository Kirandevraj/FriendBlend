import cv2
import sys
import numpy as np

def clahe(img, grid_size=7):
	# create a clahe object
	clh_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(grid_size,grid_size))
	lab_space = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
	# split image into RGB channel images
	img_channels = cv2.split(lab_space)
	img_channels[0] = clh_obj.apply(img_channels[0])
	return cv2.cvtColor(cv2.merge(img_channels), cv2.COLOR_LAB2BGR)

def haar_body_detector(img):
    min_size, max_size = 1.5, 5
    # load .xml classifier file
    cascade_obj = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray_scale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # detect different size objects and return them as rectangles
    obj_rect = cascade_obj.detectMultiScale(gray_scale, min_size, max_size)
    iterations, temp = 0, 0
    while(len(obj_rect)==0):
        if iterations>=5:
            break
        iterations += 1
        min_size -= 0.1
        obj_rect = cascade_obj.detectMultiScale(gray_scale, min_size, max_size)
    # Hamming distance used as measure for keypoint matching 
    for (x,y,w,h) in obj_rect:
        xmax = min(img.shape[1],x+int(2.2*w))
        ymax = img.shape[0]
        xmin = max(0,int(x-int(1.5*w)))
        ymin = max(0,int(y-h))
        if (ymax-ymin)*(xmax-xmin) > temp:
            bbox = [(xmin,ymin,xmax,ymax,w,h)]
            temp = (ymax-ymin)*(xmax-xmin)
    img = cv2.rectangle(img,(xmin,ymin,xmax,ymax),(255,0,0),2)
    return bbox, cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

def ordering_img(img1,img2,bbox_1,bbox_2):
    if bbox_1[0][2]>bbox_2[0][0]:
    	return img2.copy(), img1.copy(), bbox_2.copy(), bbox_1.copy()

def kp_orb_detection(img, num_kp):
	# find features using orb
	orb = cv2.ORB_create(nfeatures=num_kp).detect(img)
	return orb

def viable_kp(bbox_1,bbox_2,kp):
	img_kp = kp.copy()
	for i in kp:
		# Remove keypoint matches from bounding boxes of each subject
		if (not(((i.pt[0]<bbox_1[0][0] or i.pt[0]>bbox_1[0][2]) or (i.pt[1]<bbox_1[0][1] or i.pt[1]>bbox_1[0][3])) and ((i.pt[0]<bbox_2[0][0] or i.pt[0]>bbox_2[0][2]) or (i.pt[1]<bbox_2[0][1] or i.pt[1]>bbox_2[0][3])))):
			img_kp.remove(i)
		else:
			continue
	return img_kp

def kp_orb_descriptor(img, kp, num_kp):
	# compute the descriptors with orb
	kp, desc = cv2.ORB_create(nfeatures=num_kp).compute(img,kp)
	return desc

def kp_bf(desc1, desc2):
	num_kp=40
	# brute force matcher
	bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(desc1,desc2)
	bf = sorted(bf, key = lambda x:x.distance)
	if (500 > len(bf)):
		return bf[0:20]
	return bf[0:num_kp]

def extract_bf_matches(bf_match, kpts1, kpts2):
    dest  = np.float32([kpts2[bf_match[i].trainIdx].pt for i in range(len(bf_match))]).reshape(-1,1,2)
    src  = np.float32([kpts1[bf_match[i].queryIdx].pt for i in range(len(bf_match))]).reshape(-1,1,2)
    return src, dest

def grabcut(img1,img2,bbox_1,bbox_2):
    mask = np.zeros(img1.shape[:2],np.uint8)
    bg_model, fg_model = np.zeros((1,65),np.float64), np.zeros((1,65),np.float64)
    if bbox_1[0][4]*bbox_1[0][5] <= bbox_1[1][4]*bbox_1[1][5]:
        p=1
    elif bbox_1[0][4]*bbox_1[0][5] > bbox_1[1][4]*bbox_1[1][5]:
    	p=0
    rect = (bbox_1[p][0],bbox_1[p][1],bbox_1[p][2],bbox_1[p][3])
    for i in range(bbox_1[p][0],bbox_1[p][2]):
        for j in range(bbox_1[p][1],bbox_1[p][3]):
            mask[j,i] = cv2.GC_PR_FGD
    for i in range(bbox_1[p][0]+bbox_1[p][4]-5,bbox_1[p][2]-bbox_1[p][4]+5):
        for j in range(bbox_1[p][1]+bbox_1[p][5]-5,bbox_1[p][1]+int(1*bbox_1[p][5])+5):
            mask[j,i] = cv2.GC_FGD
    cv2.grabCut(img1.copy(), mask,rect,bg_model,fg_model,1,cv2.GC_INIT_WITH_MASK)
    return img1.copy()*np.where((mask==2)|(mask==0),0,1).astype('uint8')[:,:,np.newaxis],img2.copy()

def alpha_blending(img1, img2, bbox_1, bbox_2):
    if bbox_2[0][0] < bbox_1[0][2]:
        temp1, temp2 = img1.copy(), img2.copy()
        img1, img2 = img2.copy(), temp1.copy()
        bbox_1, bbox_2 = bbox_2.copy(), temp2.copy()
    out_img = np.zeros(img1.shape)
    step_size = 1/(bbox_2[0][0]-bbox_1[0][2])
    for x in range(bbox_1[0][2],bbox_2[0][0]):
        step_cnt = x-bbox_1[0][2]
        for i in range(3):
        	out_img[:,x,i] = ((1-(step_cnt*step_size))*img1[:,x,i])+((step_cnt*step_size)*img2[:,x,i])
    out_img[:,0:bbox_1[0][2],:] = img1[:,0:bbox_1[0][2],:]
    out_img[:,bbox_2[0][0]:,:] = img2[:,bbox_2[0][0]:,:]
    return out_img

def crop_image(img, H):
	pts = np.float32([[0, 0], [img.shape[1], 0], [img.shape[1], img.shape[0]], [0, img.shape[0]]]).reshape(-1,1,2)
	warp_points = cv2.perspectiveTransform(pts, H)
	# calculate top row and bottom row for cropped image
	return img[int(max(max(warp_points[0][0][1], warp_points[1][0][1]), 0)):int(min(min(warp_points[2][0][1], warp_points[3][0][1]), img.shape[0])), 0:img.shape[1]]

def blend_crop_img(bg_img, img):
    img_gray_cropped = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_bin_cropped = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)
    cv2.threshold(img_gray_cropped, 0, 255, cv2.THRESH_BINARY, img_bin_cropped)
    merged_img = np.uint8((255 - img_bin_cropped) / 255) * bg_img + img
    concat_image = np.concatenate((img_gray_cropped, img_bin_cropped[:,:,0]),axis=1)
    for erosion_size, fg_coeff in [(5, 0)]:
        element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_size, erosion_size))
        inner_mask = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)
        cv2.erode(img_bin_cropped, element, dst=inner_mask)
        blended_section = np.uint8(fg_coeff * (img_bin_cropped - inner_mask) / 255 * img +(1 - fg_coeff) * (img_bin_cropped - inner_mask) / 255 * bg_img)
        inverse_mask = (255 - (img_bin_cropped - inner_mask))/255  
        merged_img = inverse_mask * merged_img + blended_section
        img_bin_cropped = inner_mask
    for i in range(0, inner_mask.shape[0]):
        for j in range(0, inner_mask.shape[1]):
            if np.uint8(255) == inner_mask[i, j]:
                br = i
                break
    merged_img = merged_img[0:br, 0:inner_mask.shape[1]]
    return merged_img
