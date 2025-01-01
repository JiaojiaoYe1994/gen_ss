import os
import re
from numpy import random
# /JPEGImages/2007_000032.jpg /SegmentationClassAug/2007_000032.png
from similarity_test import calculate_similarity

similarity_threshold = 0.36 # todo, a little bit smaller than threasold
base_root = "/mnt/nas/jiaojiao/data/VOC_aug/VOC2012/JPEGImages/"
img_root = "./output/pascal_train_0618/"
output = "./train_aug_0618.txt"
ratio = 1
datasize = 10582 # Pascal training datasize
JPGimg_folder = "/JPEGImages/"
img_list = os.listdir(img_root)
# Choose synthetic image to satisfy real/fake ratio
choice_id = random.choice(len(img_list), int(datasize*ratio*2.5), replace=False)  

pattern = r'_[0-4].png'
replacement = '_new.png'
d_sum = 0.
total_num = int(datasize*ratio)
cnt = 0 
# function1: Generate list of file name for Synthetic dataset for train, val
with open("./train_aug_0618.txt", "w") as f:
    for idx, name in enumerate(img_list):
        if idx in choice_id:
#             name_new = name.replace("_*.png", "_new.png")
            # replace annotation with SAM annotation
            name_new = re.sub(pattern, replacement, name, flags=re.IGNORECASE)
            name_base = re.sub(pattern, '.png', name, flags=re.IGNORECASE)
        
            # similarity filter
            d = calculate_similarity(base_root+name_base, img_root+name)
            print(name, d)
            if d <= similarity_threshold:
                cnt += 1
                d_sum += d.item()
                f.write("/JPEGImages_0618/" + name + " /SegmentationClassAug/" + name_new + '\n')
                if cnt >= total_num:
                    break
                    
print("Average similarity: ", d_sum / int(datasize*ratio))
