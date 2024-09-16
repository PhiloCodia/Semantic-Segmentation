import os
import cv2
from util import get_detections
import numpy as np
import random

# (1) define paths
cfg_paths = 'models/mask_rcnn_inception/mask_rcnn_inception_v2_coco_2018_01_28/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt'
weights_paths = 'models/mask_rcnn_inception/mask_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb'
class_names_paths = 'models/mask_rcnn_inception/mask_rcnn_inception_v2_coco_2018_01_28/mscoco_labels.names'
img_path = './cat.png'

# (2) load image

img = cv2.imread(img_path)
H, W, C = img.shape


# (3) load model

net = cv2.dnn.readNetFromTensorflow(weights_paths, cfg_paths)

# (4) convert image

blob = cv2.dnn.blobFromImage(img)

# (5) get masks

boxes, masks = get_detections( net, blob )

# (6) draw masks
colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(90)]

empty_img = np.zeros((H, W, C))
detection_th = 0.5
for j in range(len(masks)):
    #bbox = boundinng box
    bbox =  boxes[0, 0, j]
    
    class_id = bbox[1]
    score = bbox[2]
    
    if score > detection_th :
        mask = masks[j]
        x1, y1, x2, y2 = int(bbox[3]*W), int(bbox[4]*H), int(bbox[5]*W), int(bbox[6]*H)
        
        mask = mask[int(class_id)]
        mask = cv2.resize(mask, (x2-x1, y2-y1))
        
        _, mask = cv2.threshold(mask, 0.5, 1, cv2.THRESH_BINARY)
        #mask = mask * 255
        for c in range(3): 
            empty_img[y1:y2, x1:x2, c] = mask * colors[int(class_id)][c]
          
        print(mask.shape)
        print(H, W)
        
# (7) visualization

overlay = ((0.6 * empty_img) + (0.4 * img)).astype("uint8")

cv2.imshow('mask', empty_img)

cv2.imshow('img', img)
cv2.imshow('overlay', overlay)
cv2.waitKey(0)
cv2.destroyAllWindows()   
        