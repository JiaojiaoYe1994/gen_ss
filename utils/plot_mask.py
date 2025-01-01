# !/usr/bin/env python
# coding: utf-8
#
# Author: Jiaojiao Ye
# Date:   18 June 2024
# version 2 : 10 August 2024
#            Add apply_palette for PASCAL VOC SegmentationClassAug 

import os

import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt

def plot_seg_result(img, mask, type=None, size=500, alpha=0.5, anns='mask'):
    '''
    support_image = cv2.imread('/mnt/nas/jiaojiao/data/VOCdevkit/VOC2012/JPEGImages/2007_000346.jpg', cv2.IMREAD_COLOR)
    support_image = cv2.cvtColor(support_image, cv2.COLOR_BGR2RGB)
    support_image = np.float32(support_image)
    support_label = cv2.imread('/mnt/nas/jiaojiao/data/VOCdevkit/VOC2012/SegmentationClassAug/2007_000346.png', cv2.IMREAD_GRAYSCALE)
    
    img_pre = plot_seg_result(support_image, support_label,type="red", size=None)
    rgb_array = img_pre.astype(np.uint8) 
    cv2.imwrite("support_mask.png", rgb_array )
    '''
    assert type in ['red', 'blue', 'yellow']
    if type == 'red':
        color = (255, 50, 50)     # red  (255, 50, 50) (255, 90, 90) (252, 60, 60)
    elif type == 'blue':
        color = (90, 90, 218)   # blue (102, 140, 255) (90, 90, 218) (90, 154, 218)
    elif type == 'yellow':
        color = (255, 218, 90)  # yellow
    color_scribble = (255, 218, 90) # (255, 218, 90) (0, 0, 255)

    img_pre = img.copy()

    if anns == 'mask':
        for c in range(3):
            # Pascal has 20 classes
            for i in range(21): 
                img_pre[:, :, c] = np.where(mask[:,:] == i,
                                        img[:, :, c] * (1 - alpha) + alpha * color[c],
                                        img[:, :, c])            
    elif anns == 'scribble':
        mask[mask==255]=0
        mask = mask[:,:,0]
        dilated_size = 5
        Scribble_Expert = ScribblesRobot()
        scribble_mask = Scribble_Expert.generate_scribbles(mask)
        scribble_mask = ndimage.maximum_filter(scribble_mask, size=dilated_size) # 
        for c in range(3):
            img_pre[:, :, c] = np.where(scribble_mask == 1,
                                        color_scribble[c],
                                        img[:, :, c])                    
    elif anns == 'bbox':
        mask[mask==255]=0
        mask = mask[:,:,0]        
        bboxs = find_bbox(mask)
        for j in bboxs: 
            cv2.rectangle(img_pre, (j[0], j[1]), (j[0] + j[2], j[1] + j[3]), (255, 0, 0), 4) # -1->fill; 2->draw_rec

    img_pre = cv2.cvtColor(img_pre, cv2.COLOR_RGB2BGR)  
    
    if size is not None:
        img_pre = cv2.resize(img_pre, dsize=(size, size), interpolation=cv2.INTER_LINEAR)

    return img_pre


'''
convert the segmentation masks in the Pascal VOC 2012 dataset (particularly the SegmentationClassAug directory) into a palettized format with a color palette
'''

# Define the Pascal VOC palette
VOC_PALETTE = [
    (0, 0, 0),        # 0: background
    (128, 0, 0),      # 1: aeroplane
    (0, 128, 0),      # 2: bicycle
    (128, 128, 0),    # 3: bird
    (0, 0, 128),      # 4: boat
    (128, 0, 128),    # 5: bottle
    (0, 128, 128),    # 6: bus
    (128, 128, 128),  # 7: car
    (64, 0, 0),       # 8: cat
    (192, 0, 0),      # 9: chair
    (64, 128, 0),     # 10: cow
    (192, 128, 0),    # 11: dining table
    (64, 0, 128),     # 12: dog
    (192, 0, 128),    # 13: horse
    (64, 128, 128),   # 14: motorbike
    (192, 128, 128),  # 15: person
    (0, 64, 0),       # 16: potted plant
    (128, 64, 0),     # 17: sheep
    (0, 192, 0),      # 18: sofa
    (128, 192, 0),    # 19: train
    (0, 64, 128),     # 20: tv/monitor
]

def apply_palette(mask):
    # Convert the mask to a PIL Image in 'P' mode (palettized)
    mask_pil = Image.fromarray(mask.astype(np.uint8), mode='P')
    
    # Flatten the VOC palette list and set it as the palette for the image
    flat_palette = [value for color in VOC_PALETTE for value in color]
    mask_pil.putpalette(flat_palette)
    
    return mask_pil


# print(support_image.shape, support_label.shape)
# print(support_image.max(), support_image.min(),support_label.max(), support_label.min())
# print(np.unique(support_label))
# image = Image.fromarray(img_pre.astype(np.uint8), mode="L")
# image.save("support_mask.png")
# mask_path = '/mnt/nas/jiaojiao/2024/HDMNet/dm/support_label/2007_009687.png'

# mask = np.array(Image.open(mask_path))

# # Apply the palette to the mask
# palettized_mask = apply_palette(mask)

# # Save or display the palettized image
# # palettized_mask.show()  # To display
# palettized_mask.save('output_path.png')  # To save


# Visualizing the results of few-shot segmentation is crucial for understanding how well the model generalizes to new classes
def visualize_few_shot_segmentation(support_images, support_masks, query_image, predicted_mask, ground_truth_mask=None, name="fss.png"):
    num_support = len(support_images)
    
    fig, axes = plt.subplots(1, num_support + 2, figsize=(15, 10))


    # Overlay the groundtruth mask on the support image
    for i in range(num_support):
        support_image_with_pred = support_images[i].copy()
        support_mask_rgb = np.zeros_like(support_image_with_pred)
        support_mask_rgb[support_masks[i] == 1] = [0, 255, 0]  # Red color for the predicted mask
        support_image_with_gt = Image.blend(Image.fromarray(support_images[i]), Image.fromarray(support_mask_rgb), alpha=0.5)

#         axes[i].imshow(support_image_with_gt)
#         axes[i].set_title(f"Support Image", fontsize=26, va='top')
#         axes[i].axis('off')
        fig, ax = plt.subplots()
        ax.imshow(support_image_with_gt)
        ax.axis('off')
        plt.savefig("./dm/vis_1-shot_support_{}".format(name))
    
    # Overlay the predicted mask on the query image
    query_image_with_pred = query_image.copy()
    predicted_mask_rgb = np.zeros_like(query_image_with_pred)
    predicted_mask_rgb[predicted_mask == 1] = [255, 0, 0]  # Red color for the predicted mask
    query_image_with_pred = Image.blend(Image.fromarray(query_image), Image.fromarray(predicted_mask_rgb), alpha=0.5)

#     axes[num_support].imshow(query_image_with_pred)
#     axes[num_support].set_title("Prediction", fontsize=26)
#     axes[num_support].axis('off')
    fig, ax = plt.subplots()
    ax.imshow(query_image_with_pred)
    ax.axis('off')
    plt.savefig("./dm/vis_1-shot_predict_{}".format(name))
    

    if ground_truth_mask is not None:
        # Overlay the ground truth mask on the query image
        query_image_with_gt = query_image.copy()
        ground_truth_mask_rgb = np.zeros_like(query_image_with_gt)
        ground_truth_mask_rgb[ground_truth_mask == 1] = [0, 255, 0]  # Green color for the ground truth mask
        query_image_with_gt = Image.blend(Image.fromarray(query_image), Image.fromarray(ground_truth_mask_rgb), alpha=0.5)

#         axes[num_support + 1].imshow(query_image_with_gt)
#         axes[num_support + 1].set_title("Ground Truth", fontsize=26)
#         axes[num_support + 1].axis('off')
        fig, ax = plt.subplots()
        ax.imshow(query_image_with_gt)
        ax.axis('off')
        
        plt.savefig("./dm/vis_1-shot_gt_{}".format(name))

#     plt.show()
#     plt.close()
    
#     plt.savefig("./dm/vis_1-shot_gt_{}".format(name))


def main():
    
    folder = '/mnt/nas/jiaojiao/2024/HDMNet/dm/query_label/'
    filelist = os.listdir('/mnt/nas/jiaojiao/2024/HDMNet/dm/query_label/')
#     mask_path = '/mnt/nas/jiaojiao/2024/HDMNet/dm/support_label/2007_009687.png'
    
    for mask_path in filelist:
        mask = np.array(Image.open(folder + mask_path))

        # Apply the palette to the mask
        palettized_mask = apply_palette(mask)

        # Save or display the palettized image
        # palettized_mask.show()  # To display
        palettized_mask.save('./dm/out/'+ mask_path)  # To save
        

# if __name__ == '__main__':
#     main()
