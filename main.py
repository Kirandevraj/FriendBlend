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
