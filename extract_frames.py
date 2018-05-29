import os
import math
import cv2
from dates import classes, data_dir

for class_name in classes:
    class_path = data_dir / class_name
    test_dir = data_dir / 'test3' / class_name
    train_dir = data_dir / 'train3' / class_name
    valid_dir = data_dir / 'valid3' / class_name
    test_dir.mkdir(parents=True, exist_ok=True)
    train_dir.mkdir(parents=True, exist_ok=True)

    videos = os.listdir(str(class_path))
    count = 0
    img_ctr = 0
    for video_name in videos:
        video = cv2.VideoCapture(str(class_path / video_name))
        framerate = video.get(5)
        dir_name = 'video_' + str(count)
        if count%10 == 0:
            dir_path = test_dir #/ dir_name
        elif count%10 == 1 or count%10 == 2:
            dir_path = valid_dir #/ dir_name
        else:
            dir_path = train_dir #/ dir_name
        dir_path.mkdir(parents=True, exist_ok=True)

        while (video.isOpened()):
            frameId = video.get(1)
            success, image = video.read()
            if not success:
                break
            image=cv2.resize(image,(224,224), interpolation = cv2.INTER_AREA)
            if (frameId % math.floor(framerate) == 0):
                
                image_name =  "image_" + str(img_ctr) + ".jpg"
                img_ctr += 1
                filepath = dir_path / image_name
                cv2.imwrite(str(filepath),image)
        video.release()
        print(class_path/video_name)
        count+=1
