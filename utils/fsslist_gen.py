import os
import re

mode = 'train'
split = 3
input_path = "./fss_list_old/{}/data_list_{}.txt".format(mode, split)
output_path = "./fss_list/{}/data_list_{}.txt".format(mode, split)

pattern = r'/mnt/proj76/bhpeng22/githubProjects/fewshot_segmentation/data/coco/{}2014/'.format(mode, mode) #val: COCO_{}2014_
replacement = '/mnt/nas/jiaojiao/data/coco2014/images/{}2014/'.format(mode)
anno_pattern = r'/mnt/proj76/bhpeng22/githubProjects/fewshot_segmentation/data/coco/annotations/{}2014/'.format(mode)
anno_replacement = '/mnt/nas/jiaojiao/data/base_annotation/coco/{}/'.format(mode)

# function1: Generate split path for dataset
with open(input_path, "r") as f:
    with open(output_path, "w") as f_new:
        for line in f:
            line_new = re.sub(pattern, replacement, line, flags=re.IGNORECASE)
            line_new = re.sub(anno_pattern, anno_replacement, line_new, flags=re.IGNORECASE)
            
#             line_new = line.replace(pattern, replacement)            
            print(line, line_new)
            f_new.write(line_new)