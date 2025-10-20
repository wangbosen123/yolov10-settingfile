from asyncore import write
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
import random
random.seed(42)

def image_and_coordinate_json_to_txt():

    for filename in os.listdir(image_path):
        json_data = json_path + f'{filename[0:-4]}.json'
        try:
            with open(json_data, 'r') as f:
                data = json.load(f)

            shapes = data['shapes']
            image_height = data['imageHeight']
            image_width = data['imageWidth']

            for info in shapes:
                points = info['points']
                coordinates = []

                for p in points:
                    coordinates.append([])
                    coordinates[-1].append(p[0]/image_width)
                    coordinates[-1].append(p[1]/image_height)


                with open(label_path + f'{filename[0:-4]}.txt', 'a') as file:
                    file.write(f'0')
                    for coor in coordinates:
                        for i in range(2):
                            file.write(f' {coor[i]}')
                    file.write('\n')
        except:
            pass


# def data_processing(h_min, h_max, s_min, s_max, v_min, v_max):
#     for filename in os.listdir(image_path):
#         image = cv2.imread(image_path + filename)
#         hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#
#         lower = np.array([h_min, s_min, v_min])
#         upper = np.array([h_max, s_max, v_max])
#
#         mask = cv2.inRange(hsv, lower, upper)
#         mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
#         cv2.imwrite(f'{image_path}{filename[0:-4]}_mask.bmp', mask)


def Yolov8_seg_train():
    model = YOLO(model_yaml).load(weight_path)
    # imgsz_list = [1280, 1920, 2500]  # 解析度遞增順序
    # epochs_list = [600, 600, 600]  # 每階段訓練 epoch
    # batch_size_list = [16, 12, 8]
    # # 執行多階段訓練
    # for i, (imgsz, epochs, batch_size) in enumerate(zip(imgsz_list, epochs_list, batch_size_list)):
    #     stage_name = f"stage_{i+1}_{imgsz}"
    #
    #     if i > 0:
    #         # 使用上一階段訓練的權重
    #         weights_path = os.path.join(model.trainer.save_dir, "weights", "last.pt")
    #         print(f"\n[INFO] Loading previous weights from: {weights_path}")
    #         model = YOLO(weights_path)
    #
    #     print(f"\n🚀 Starting training: Stage {i+1} | imgsz={imgsz}, epochs={epochs}")
    #
    #     model.train(
    #         data=data_yaml,
    #         epochs=epochs,
    #         imgsz=imgsz,
    #         batch=batch_size,
    #         device=device,
    #         name=stage_name,
    #         rect=False,
    #     )
    #
    # print("\n✅ 多階段訓練完成！")
    model.train(data=data_yaml, epochs=epochs, batch=batch_size, device=device, imgsz=imgsz, rect=False, val_period=1, name=name)


def Yolov8_seg_inference():
    # inference_images = []
    # for filename in os.listdir(inference_path):
    #     if '.jpg' not in filename: continue
    #     image = cv2.imread(inference_path + filename)
    #     inference_images.append(image)
    # model = YOLO(weight_path)
    # model.predict(inference_images, save=True, save_txt=save_txt, show_labels=show_labels, show_conf=show_conf,
    #               show_boxes=show_boxes, conf=conf, iou=iou, exist_ok=exist_ok)

    model = YOLO(weight_path)
    for num, filename in enumerate(os.listdir(inference_path)):
        if '.jpg' not in filename: continue
        image = cv2.imread(inference_path + filename)
        results = model.predict(
            source=image,
            save=True,
            save_txt=True,
            project=save_dir,
            name=str(num),
            exist_ok=True,
            show_boxes=True,
            conf=0.3,
            iou=0.5,
            show_conf=True
        )


if __name__ == '__main__':
    with open('yolov8-seg.yaml', 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    if config['Data_preprocess']['default']:
        image_path = config['Data_preprocess']['image_path']
        label_path = config['Data_preprocess']['label_path']
        json_path = config['Data_preprocess']['json_path']
        train_number = config['Data_preprocess']['train_number']
        yolov8_seg_path = config['Data_preprocess']['yolov8_seg_path']
        image_and_coordinate_json_to_txt()


    if config['Train_yolov8_seg']['default']:
        device = config['Train_yolov8_seg']['device']
        os.environ['CUDA_DEVICES_ORDER'] = 'PIC_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = device

        epochs = config['Train_yolov8_seg']['epochs']
        batch_size = config['Train_yolov8_seg']['batch_size']
        imgsz = config['Train_yolov8_seg']['imgsz']
        model_yaml = config['Train_yolov8_seg']['model_yaml']
        weight_path = config['Train_yolov8_seg']['weight_path']
        data_yaml = config['Train_yolov8_seg']['data_yaml']
        name = config['Train_yolov8_seg']['name']

        Yolov8_seg_train()


    if config['Test_yolov8_seg']['default']:
        os.environ['CUDA_DEVICES_ORDER'] = 'PIC_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
        inference_path = config['Test_yolov8_seg']['inference_path']
        weight_path = config['Test_yolov8_seg']['weight_path']

        save_txt = config['Test_yolov8_seg']['save_txt']
        show_labels = config['Test_yolov8_seg']['show_labels']
        show_conf = config['Test_yolov8_seg']['show_conf']
        show_boxes = config['Test_yolov8_seg']['show_boxes']
        conf = config['Test_yolov8_seg']['conf']
        iou = config['Test_yolov8_seg']['iou']
        exist_ok = config['Test_yolov8_seg']['exist_ok']
        save_dir = config['Test_yolov8_seg']['save_dir']

        Yolov8_seg_inference()











