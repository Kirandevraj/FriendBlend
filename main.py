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

print(np.shape(Image_1),args.op)
cv2.imwrite(args.op,Image_1)
