import os

# /JPEGImages/2007_000032.jpg /SegmentationClassAug/2007_000032.png

label_root = "/mnt/nas/jiaojiao/data/VOC_aug/VOC2012/SegmentationClassAug"
img_root = "/mnt/nas/jiaojiao/data/VOC_aug/VOC2012/JPEGImages"

img_list = os.listdir(img_root)

# print(img_list)
print(len(img_list))

# function1: Generate list of file name for Synthetic dataset
# with open("./train_aug.txt", "w") as f:
#     for name in img_list:
#         f.write("/JPEGImages/" + name + " /SegmentationClassAug/" + name + '\n')
        
    
# Function2: Replace mask in train dataset with SAM segmented semantics
with open("./train_aug.txt", "w") as f:

    # Open the file in read mode
    with open('./train_aug_old.txt', 'r') as file:
        # Read all lines into a list
        lines = file.readlines()
        print(lines)
        # Iterate over the list
        for line in lines:

            # replace mask in train dataset with SAM segmented semantics
            line_old = line.strip()
            line_new = line_old.replace(".png", "_new.png" )
#             print(line_old)
#             print(line_new)
            f.write(line_new + '\n')

            # Print each line
    #        print(line.strip())
