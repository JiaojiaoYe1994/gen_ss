import os
import re
from numpy import random
# /JPEGImages/2007_000032.jpg /SegmentationClassAug/2007_000032.png

img_root = "./output/pascal_train"
ratio = 1
datasize = 10582 # Pascal training datasize
JPGimg_folder = "/JPEGImages/"
img_list = os.listdir(img_root)
choice_id = random.choice(len(img_list), datasize*ratio, replace=False) # Choose synthetic image to satisfy real/fake ratio
print("Length of image in dataset"len(img_list))

pattern = r'_[0-4].png'
replacement = '_new.png'
# function1: Generate list of file name for Synthetic dataset
with open("./train_aug.txt", "w") as f:
    for idx, name in enumerate(img_list):
        if idx in choice_id:
           name_new = name.replace("_*.png", "_new.png")
           name_new = re.sub(pattern, replacement, name, flags=re.IGNORECASE)
           f.write("/JPEGImages/" + name + " /SegmentationClassAug/" + name + '\n')
        
