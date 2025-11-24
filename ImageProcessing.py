import cv2 
import os 
from pathlib import Path
import shutil
from torch.utils.data import random_split
import torch
import sys 
torch.manual_seed = 42

def Getlabel(data_path, img_path, label):
    CNNs = os.path.join(data_path, f'{label}_processed_CNNs')
    YOLO = os.path.join(data_path, f'{label}_processed_YOLO')    

    for dir_path in [CNNs, YOLO]:
        if not os.path.exists(dir_path): 
            os.makedirs(dir_path) 
            print(f'Folder is created at: {dir_path}.')

    img_list = [path for path in Path(img_path).rglob('*')]
    print(f'len(img_list): {len(img_list)}')
    
    for img_path in img_list:
        boxes = []
        print(img_path)
        while True:
            img = cv2.imread(img_path)

            h, w = img.shape[:2]
            if h > 650: 
                print(img.shape)
                img = cv2.resize(img, (488, 650))

            box = list(cv2.selectROI('Select ROI', img, fromCenter=False, showCrosshair=False))
            print(box)
            if box[2] > 0 and box[3] > 0:
                boxes.append(box)
            key = cv2.waitKey(500) & 0xFF
            if key == ord('c'):
                break
        boxes_str = ''
        for box in boxes:
            # YOLO
            boxes_str = boxes_str + str(box[0]) + '_' + str(box[1]) + '_' + str(box[3]) + '_' + str(box[3]) + '_'
            center_x = int((box[0] + box[2])/2)
            center_y = int((box[1] + box[2])/2)
            boxes_str = boxes_str + str(center_x) + '_' + str(center_y) +'_'
            # CNNs
            cropped_img = img[box[1]:(box[1]+box[3]), box[0]:(box[0]+box[2])]
            cropped_img_name = f'{label}_{box[0]}_{box[1]}_{box[2]}_{box[3]}.jpg'
            cv2.imwrite(cropped_img_name, cropped_img)
            shutil.move(f'D:/DoAn/{cropped_img_name}', os.path.join(CNNs,cropped_img_name))
        
        new_image_name = label + boxes_str 
        new_image_name = new_image_name[:-1] + '.jpg'
        
        if (len(new_image_name) > (len(label)+4)):
            shutil.copy(img_path, os.path.join(YOLO, new_image_name))
            
        img_list.remove(img_path)
        if os.path.exists(img_path):
            os.remove(img_path)
        print(len(img_list))

def cropImage (video_path, image_path): # from videos
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Cannot open")
        exit()

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_count = 0
    saved_count = 0
    while True: 
        ret, frame = cap.read()
        if not ret:
            break 

        if frame_count % fps == 0:   
            filename = f'frame_{saved_count}.jpg'
            cv2.imwrite(filename, frame)
            shutil.move(filename, image_path)
            saved_count += 1

        frame_count += 1
    
    cap.release()
    print('Finished capturing')

def resizeImg(img_path):
    img_list = [path for path in Path(img_path).rglob('*')]
    for path in img_list:
        img = cv2.imread(path)
        print(path)
        img = cv2.resize(img,(240,240))
        cv2.imwrite("res.jpg", img) 
        shutil.copy('res.jpg', path)
        if os.path.exists('res.jpg'):
            os.remove('res.jpg')

def moveToDirectory(path_list, target_folder):
    for path in path_list:
        img_name = path.name
        if not os.path.exists(os.path.join(target_folder, img_name)):
            shutil.move(path, target_folder)

def splitFile(img_path, data_path_train, data_path_val, label):
    img_train_folder = os.path.join(data_path_train, label)
    img_val_folder = os.path.join(data_path_val, label)

    if not os.path.exists(img_train_folder):
        os.mkdir(img_train_folder)

    if not os.path.exists(img_val_folder):
        os.mkdir(img_val_folder)

    file_list = [path for path in Path(img_path).rglob('*')]
    train_number = int(len(file_list) * 0.9)
    train, val = random_split(file_list, [train_number, len(file_list)-train_number])

    moveToDirectory(train, img_train_folder)
    moveToDirectory(val, img_val_folder)

def RenameFile(data_path, old_label, new_label):
    path_list = [path for path in Path(data_path).rglob('*')]

    for path in path_list:
        if path.name.startswith(old_label):
            p = Path(path)
            file_name = path.name
            new_file_name =  file_name.replace(old_label, new_label)
            new_path_name = os.path.join(p.parent, new_file_name)
            os.rename(path, new_path_name)
    
    print('Finished renaming.')


# # LABEL
data_path = 'D:/DoAn'
img_path = r'D:\DoAn\bimbim'
# list_img = [path for path in Path(img_path).rglob('*')]
# print(len(list_img))

# Getlabel(data_path, img_path, 'bimbim')

# CAPTURE IMAGES FROM VIDEOS

# video_path = r'D:\DoAn\plasticbag\7043367062966.mp4'
# img_path = r'D:\DoAn\plasticbag'

# cropImage(video_path, img_path)

# RESIZE
# img_path = 'D:/DoAn/bimbim_processed_CNNs'
# resizeImg(img_path)

# # SPLIT FILE
img_path = r'D:\DoAn\bimbim_processed_CNNs'
data_path_train = r'D:\DoAn\orange_egg_banana_train'
data_path_val = r'D:\DoAn\orange_egg_banana_val'
splitFile(img_path, data_path_train, data_path_val, 'bimbim')

# RENAME FILES
# data_path = r'D:\DoAn\plasticbag_processed_CNNs'
# RenameFile(data_path, 'cucumberpeel', 'plasticbag')


