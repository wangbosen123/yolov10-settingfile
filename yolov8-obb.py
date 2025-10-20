from ultralytics import YOLOv10
from ultralytics import YOLO
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
import json
import shutil
from PIL import Image
from PIL import ImageFile

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True


def image_and_coordinate_flip_json_to_txt():

    for filename in os.listdir(image_path):
        image = cv2.imread(image_path + filename)
        image_mirror = cv2.flip(image, 1)
        json_data = json_path + f'{filename[0:-4]}.json'
        with open(json_data, 'r') as f:
            data = json.load(f)

        shapes = data['shapes']
        image_height = data['imageHeight']
        image_width = data['imageWidth']

        for info in shapes:
            label = info['label']
            points = info['points']

            p1_init, p4_init = [points[0][0]/image_width, points[0][1]/image_height], [points[1][0]/image_width, points[1][1]/image_height]
            p2_init, p3_init = [p4_init[0], p1_init[1]], [p1_init[0], p4_init[1]]

            p1 = [min(p1_init[0], p2_init[0], p3_init[0], p4_init[0]), min(p1_init[1], p2_init[1], p3_init[1], p4_init[1])]
            p2 = [max(p1_init[0], p2_init[0], p3_init[0], p4_init[0]), min(p1_init[1], p2_init[1], p3_init[1], p4_init[1])]
            p3 = [max(p1_init[0], p2_init[0], p3_init[0], p4_init[0]), max(p1_init[1], p2_init[1], p3_init[1], p4_init[1])]
            p4 = [min(p1_init[0], p2_init[0], p3_init[0], p4_init[0]), max(p1_init[1], p2_init[1], p3_init[1], p4_init[1])]


            with open(restore_txt_path + f'{filename[0:-4]}.txt', 'a') as file:
                file.write(f'0')
                for i in range(2):file.write(f' {p1[i]}')
                for i in range(2):file.write(f' {p2[i]}')
                for i in range(2):file.write(f' {p3[i]}')
                for i in range(2):file.write(f' {p4[i]}')
                file.write('\n')
            cv2.imwrite(f'{restore_image_path}{filename[0:-4]}.jpg', image)


            if mirror:
                p1_mirror, p4_mirror = [1 - p1[0], p1[1]], [1 - p4[0], p4[1]]
                p2_mirror, p3_mirror = [1 - p2[0], p2[1]], [1 - p3[0], p3[1]]
                with open(restore_txt_path + f'{filename[0:-4]}_mirror.txt', 'a') as file:
                    file.write(f'{label}')
                    for i in range(2): file.write(f' {p1_mirror[i]}')
                    for i in range(2): file.write(f' {p2_mirror[i]}')
                    for i in range(2): file.write(f' {p3_mirror[i]}')
                    for i in range(2): file.write(f' {p4_mirror[i]}')
                    file.write('\n')
                cv2.imwrite(f'{restore_image_path}{filename[0:-4]}_mirror.jpg', image_mirror)


def split_data():
    for num, filename in enumerate(os.listdir(restore_image_path)):
        if num < train_number:
            target_path = yolov8_data_path + 'train/'
        else:
            target_path = yolov8_data_path + 'val/'

        image_path = restore_image_path + filename
        txt_path = restore_txt_path + filename[0:-4] + '.txt'

        shutil.copy(image_path, target_path+ 'images/')
        shutil.copy(txt_path, target_path + 'labels/')


def sliding_window():
    w, h = image.shape[1], image.shape[0]
    for row in range(0, w, window_size):
        for col in range(0, h, window_size):
            crop_image = image[row: (row+window_size), col: (col+window_size)]
            crop_image = cv2.cvtColor(crop_image, cv2.COLOR_BGR2RGB)
            cv2.imwrite(f'datasets/surface-obb/Images/Sample{sample_number}/Row{row}_Col{col}_Size{window_size}.bmp', crop_image)


def YOLO_json_to_txt():
    for image_filename in os.listdir(image_path):
        if not (image_filename.endswith('bmp')):
            continue

        base_name = os.path.splitext(image_filename)[0]
        json_file = os.path.join(json_path, base_name + '.json')
        txt_file = os.path.join(restore_txt_path, base_name + '.txt')

        if os.path.exists(json_file):
            with open(json_file, 'r') as f:
                data = json.load(f)

            image_width = data['imageWidth']
            image_height = data['imageHeight']
            shapes = data['shapes']
            label = 0  # 瑕疵的類別標籤

            with open(txt_file, 'w') as f_txt:  # 直接 w 寫入，避免append造成重複
                for shape in shapes:
                    norm_points = []
                    points = shape['points']
                    for pt in points:
                        x_norm = pt[0] / image_width
                        y_norm = pt[1] / image_height
                        norm_points.append((x_norm, y_norm))

                    f_txt.write(f"{label}")
                    for x, y in norm_points:
                        f_txt.write(f" {x} {y}")
                    f_txt.write('\n')
        else:
            # JSON 不存在，產生空txt檔
            open(txt_file, 'w').close()

        # for filename in os.listdir(json_path):
        #     if not filename.endswith('.json'): continue
        # json_file = os.path.join(json_path, filename)
        # with open(json_file, 'r') as f:
        #     data = json.load(f)
        #
        # image_width = data['imageWidth']
        # image_height = data['imageHeight']
        # shapes = data['shapes']
        # label = 0 # Always define the label of surface defect is 0
        #
        # for shape in shapes:
        #     norm_points = []
        #     points = shape['points']
        #     for pt in points:
        #         x_norm = pt[0] / image_width
        #         y_norm = pt[1] / image_height
        #         norm_points.append((x_norm, y_norm))
        #
        #     txt_filename = os.path.join(restore_txt_path, filename.replace('.json', '.txt'))
        #     with open(txt_filename, 'a') as f_txt:
        #         f_txt.write(f"{label}")
        #         for x, y in norm_points:
        #             f_txt.write(f" {x} {y}")
        #         f_txt.write('\n')


def Yolov8_obb_train():
    model = YOLO(model_yaml).load(weight_path)
    model.train(data=data_yaml, epochs=epochs, batch=batch_size, device=device)


def Yolov8_obb_inference():
    inference_images = []
    for filename in os.listdir(inference_path):
        image = cv2.imread(inference_path + filename)
        inference_images.append(image)

    model = YOLO(weight_path)
    model.predict(inference_images, save=True, save_txt=save_txt, show_labels=show_labels, show_conf=show_conf,
                  show_boxes=show_boxes, conf=conf, iou=iou, exist_ok=exist_ok)



if __name__ == '__main__':
    with open('yolov8-obb.yaml', 'r') as file:
        config = yaml.load(file, Loader = yaml.FullLoader)

    if config['Sliding_Window']['default']:
        window_size = config['Sliding_Window']['window_size']
        Full_Image_Path = config['Sliding_Window']['Full_Image_Path']
        sample_number = config['Sliding_Window']['sample_number']

        image = Image.open(Full_Image_Path)
        image = np.array(image)
        image = cv2.resize(image, (12800, 12800), cv2.INTER_CUBIC)
        sliding_window()


    if config['Data_preprocess']['default']:
        image_path = config['Data_preprocess']['image_path']
        json_path = config['Data_preprocess']['json_path']
        restore_image_path = config['Data_preprocess']['restore_image_path']
        restore_txt_path = config['Data_preprocess']['restore_txt_path']
        yolov8_data_path = config['Data_preprocess']['yolov8_data_path']
        train_number = config['Data_preprocess']['train_number']
        mirror = config['Data_preprocess']['mirror']

        YOLO_json_to_txt()


    if config['Train_yolov8_obb']['default']:
        device = config['Train_yolov8_obb']['device']
        os.environ['CUDA_DEVICES_ORDER'] = 'PIC_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = device

        epochs = config['Train_yolov8_obb']['epochs']
        batch_size = config['Train_yolov8_obb']['batch_size']
        model_yaml = config['Train_yolov8_obb']['model_yaml']
        weight_path = config['Train_yolov8_obb']['weight_path']
        data_yaml = config['Train_yolov8_obb']['data_yaml']

        Yolov8_obb_train()


    if config['Test_yolov8_obb']['default']:
        inference_path = config['Test_yolov8_obb']['inference_path']
        weight_path = config['Test_yolov8_obb']['weight_path']

        save_txt = config['Test_yolov8_obb']['save_txt']
        show_labels = config['Test_yolov8_obb']['show_labels']
        show_conf = config['Test_yolov8_obb']['show_conf']
        show_boxes = config['Test_yolov8_obb']['show_boxes']
        conf = config['Test_yolov8_obb']['conf']
        iou = config['Test_yolov8_obb']['iou']
        exist_ok = config['Test_yolov8_obb']['exist_ok']

        Yolov8_obb_inference()













