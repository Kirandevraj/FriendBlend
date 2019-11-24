import cv2
import sys
import argparse
from func import *

parser = argparse.ArgumentParser()
parser.add_argument("--img1", help="Path for img1")
parser.add_argument("--img2", help="Path for img2")
parser.add_argument("--method", help="Blending method", default="automatic", choices=["automatic", "grabcut", "alphablend"])
parser.add_argument("--out", help="Path for result", default="Results/out.jpg")
args = parser.parse_args()

num_kp = 1000
img1, img2 = cv2.imread(args.img1), cv2.imread(args.img2) 
img2 = cv2.resize(img2,(img1.shape[1],img1.shape[0]))

#  contrast  limited  histogram  equalization  in  the  Lab color space on the lightness channel to preserve hue
img1, img2 = clahe(img1), clahe(img2)

# return bounding box for face and body of subject
bbox_1, img_bbox_1 = haar_body_detector(img1.copy())
if len(bbox_1) == 0:
	raise Exception("Face not detected in first input image")
	sys.exit()
bbox_2, img_bbox_2 = haar_body_detector(img2.copy())
if len(bbox_1) == 0:
	raise Exception("Face not detected in second input image")
	sys.exit()

# order images
img1,img2,bbox_1,bbox_2 = ordering_img(img1,img2,bbox_1,bbox_2)

# determine mapping between the 2 input images
kp_1, kp_2 = kp_orb_detection(img1,num_kp), kp_orb_detection(img2,num_kp)

# find valid keypoints in an image i.e not present on the bounding boxes
kp1_viable, kp2_viable = viable_kp(bbox_1,bbox_2,kp_1), viable_kp(bbox_1,bbox_2,kp_2)

# compute descriptors using orb
desc_1  = kp_orb_descriptor(img1,kp1_viable, num_kp) 
desc_2 = kp_orb_descriptor(img2,kp2_viable, num_kp)

# brute force matching using hamming distance as the measure
kp_bf_match = kp_bf(desc_1, desc_2)

# extract the brute force matches
src_pts, dest_pts = extract_bf_matches(kp_bf_match, kp1_viable, kp2_viable)

# find the transform between matched keypoints
hgraph_mat, status = cv2.findHomography(src_pts, dest_pts)

# change the apparent perspective of an image using geometric transformation
img_src = img1.copy()
warp_img = cv2.warpPerspective(img_src, hgraph_mat, (img_src.shape[1],img_src.shape[0]))

# tl_x1,tl_y1,br_x1,br_y1
orig_pt = np.float32([[[bbox_1[0][0], bbox_1[0][1]]],[[bbox_1[0][2], bbox_1[0][1]]],[[bbox_1[0][0], bbox_1[0][3]]] ,[[bbox_1[0][2],bbox_1[0][3]]]])

# compute the perspective transform pertr_pts
pertr_pts = cv2.perspectiveTransform(orig_pt, hgraph_mat)
pertr_pts[pertr_pts < 0] = 0
pertr_pts = pertr_pts.reshape((pertr_pts.shape[0], pertr_pts.shape[2]))
pertr_pts = pertr_pts.astype(int)
p1,p2 = pertr_pts[0]
p3,p4 = pertr_pts[-1]
w, h = bbox_1[0][4], bbox_1[0][5]

# decide approach based on distance between the bounding boxes
if args.method == "automatic":
	if bbox_2[0][2]-bbox_1[0][2]<200:
		args.method="grabcut"
	else:
		args.method="alphablend"

# perform alphablend and grabcut
if args.method=="alphablend":
	print("Implementing Alpha Blending as bodies are far enough")
	out_img = alpha_blending(warp_img,img2,[(p1,p2,p3,p4)],bbox_2)
	out_img = crop_image(out_img, hgraph_mat)
elif args.method=="grabcut":
	print("Implementing GrabCut as bodies are close enough")
	grabcut_img,bg = grabcut(warp_img,img2,[(p1,p2,p3,p4,w,h)],bbox_2)
	out_img = blend_crop_img(bg,grabcut_img)
	out_img = crop_image(out_img, hgraph_mat)	
	# Where nothing works, gaussian filter does :)
	out_img = cv2.GaussianBlur(out_img,(3,3),0)

cv2.imwrite(args.out, out_img)
