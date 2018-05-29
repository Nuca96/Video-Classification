import numpy as np
import os
import pandas as pd
import cv2
from scipy.misc import imread,imresize
import pickle

from dates import classes, data_dir


for method in ['train', 'test', 'valid']:
    videos = []
    labels = []
    count = 0

    for video_class in classes:
        print(video_class)
        class_dir = data_dir / method / video_class
        childs = os.listdir(str(class_dir))
        for video_name in childs:
            video_path = class_dir / video_name
            img_names = os.listdir(str(video_path))
            images = []
            lab = []
            for image_name in img_names:
                image_path = video_path / image_name
                image = imread(str(image_path))
                image = imresize(image , (224,224))
                images.append(image)
                lab.append(video_class)
                count+=1
            videos.extend(images)
            labels.extend(lab)

    x = np.array(videos)
    y = np.array(labels)
    np.save(str(data_dir / (method + '_images2')), x)
    np.save(str(data_dir / (method + '_labels2')), y)


