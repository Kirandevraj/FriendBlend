import cv2
import sys
import argparse
from func import *


parser = argparse.ArgumentParser()
parser.add_argument("--inp1", help="Path to input image 1")
parser.add_argument("--inp2", help="Path to input image 2")
parser.add_argument("--op", help="Path for the result to be stored", default="Results/op.jpg")
parser.add_argument("--technique", help="Which blending technique to use", default="auto", choices=["auto", "grabcut", "alphablend"])
args = parser.parse_args()

n_keypoints = 10000
Image_1 = cv2.imread(args.inp1)
Image_2 = cv2.imread(args.inp2)

x,y,_ = Image_1.shape
Image_2 = cv2.resize(Image_2,(y,x))

Image_1 = lab_contrast(Image_1)
Image_2 = lab_contrast(Image_2)

body_1, i_b1 = detect_body(Image_1.copy())
body_2, i_b2 = detect_body(Image_2.copy())

if (len(body_1) == 0 or len(body_2) == 0):
    print("Face not detected in one/both Images")
    sys.exit()

keypoints_valid_1 = valid_keypoints(body_1,body_2,keypoints_1)
keypoints_valid_2 = valid_keypoints(body_1,body_2,keypoints_2)

_, descriptor1  = keypoints_orb_descriptor(Image_1,keypoints_valid_1, n_keypoints)
_, descriptor2  = keypoints_orb_descriptor(Image_2,keypoints_valid_2, n_keypoints)

keypoint_matches = keypoint_bf_matcher(descriptor1, descriptor2)

print(np.shape(Image_1),args.op)
cv2.imwrite(args.op,Image_1)
doesnhomography_matrix = calculate_homography_matrix(source_points, destination_points)

homography_warped_1 = warp_perspective(Image_1.copy(), homography_matrix)

top_left_x1,top_left_y1,bot_right_x1,bot_right_y1,w,h=body_1[0]
pt1 = np.float32([[[top_left_x1, top_left_y1]],[[bot_right_x1, top_left_y1]],[[top_left_x1, bot_right_y1]] ,[[bot_right_x1,bot_right_y1]]])
