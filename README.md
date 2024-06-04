# gen_ss

Step 1
`test.ipynb`: Seg image with SAM and GroundDino

Step 2
`generate.py`  : Given mask and prompt, script to generate synthetic image with 'ControlNet' pretrained on ADE20K.

Step 3
`deeplab-python` train (optional: weakly supervised/ Few-shot) segmentation network with synthetic images. 

