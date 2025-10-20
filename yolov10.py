from ultralytics import YOLOv10
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml


class YOLOv10_method():
    def __init__(self, train_path, val_path, test_path):
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path


    def load_inference_image(self):
        inference_image_path = self.val_path + 'images/'
        images = []
        for filename in os.listdir(inference_image_path):
            image = cv2.imread(inference_image_path + filename)
            images.append(image)
        return images


    def Train_YOLOv10(self, model_yaml_path, data_yaml_path, pre_model_name, epochs, batch, device):
        model = YOLOv10(model_yaml_path).load(pre_model_name)
        model.train(data=data_yaml_path, epochs=epochs, batch=batch, device=device)
        return


    def inference(self, weights, save_txt, save_crop, show_labels, show_conf, show_boxes, conf, iou, exist_ok):
        inference_images = self.load_inference_image()
        model = YOLOv10(f'runs/detect/{weights}/weights/best.pt')
        model.predict(inference_images, save=True, save_txt=save_txt, save_crop=save_crop, show_labels=show_labels, show_conf=show_conf,
                      show_boxes=show_boxes, conf=conf, iou=iou, exist_ok=exist_ok)
        return


    def single_image_txt_to_image(self, single_txt_file, single_image_file):
        image = cv2.imread(single_image_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        with open(single_txt_file, 'r') as f:
            file = f.read()
            file = file.split('\n')

        h, w = image.shape[0], image.shape[1]

        for info in file:
            if info == '': continue
            info = info.split(' ')
            label, x1_, y1_, w1_, h1_ = float(info[0]), float(info[1]), float(info[2]), float(info[3]), float(info[4])
            x1_1 = w * x1_ - 0.5 * w * w1_
            x1_2 = w * x1_ + 0.5 * w * w1_
            y1_1 = h * y1_ - 0.5 * h * h1_
            y1_2 = h * y1_ + 0.5 * h * h1_
            cv2.rectangle(image, (int(x1_1), int(y1_1)), (int(x1_2), int(y1_2)), color=(255, 0, 0), thickness=10)
            # if label == 1: cv2.rectangel(image, (int(x1_1), int(y1_1)), (int(x1_2), int(y1_2)), color=(0, 0, 255), thickness=10)
        plt.imshow(image)
        plt.savefig('test.jpg')
        plt.close()


    def crop_yolo_detection(self, image_path, label_path, restore_path):

        if os.path.exists(restore_path+'GT_crop/'): pass
        else: os.makedirs(restore_path+'GT_crop/')

        for filename in os.listdir(image_path):
            if filename == 'labels' or filename == 'images' or filename == 'crops': continue
            title = filename.split('.')[0]
            image = cv2.imread(image_path + filename)
            label_txt = label_path + f'/{title}.txt'

            with open(label_txt, 'r') as f:
                file = f.read()
                file = file.split('\n')
            h, w = image.shape[0], image.shape[1]
            for num, info in enumerate(file):
                if info == '': continue
                info = info.split(' ')
                label, x1_, y1_, w1_, h1_ = float(info[0]), float(info[1]), float(info[2]), float(info[3]), float(info[4])
                x1_1 = w * x1_ - 0.5 * w * w1_
                x1_2 = w * x1_ + 0.5 * w * w1_
                y1_1 = h * y1_ - 0.5 * h * h1_
                y1_2 = h * y1_ + 0.5 * h * h1_
                crop_image = image[int(y1_1): int(y1_2), int(x1_1): int(x1_2)]
                print(crop_image.shape)
                print(restore_path+f'GT_crop/{filename}_{num}.jpg')
                cv2.imwrite(restore_path+f'GT_crop/{filename}_{num}.jpg', crop_image)




if __name__ == '__main__':
    with open('yolov10.yaml', 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    model_yaml_path = config['model_yaml_path']
    data_yaml_path = config['data_yaml_path']
    pre_model_name = config['pre_model_name']
    epochs = config['epochs']
    batch = config['batch']
    device = config['device']
    train_path = config['train_path']
    val_path = config['val_path']
    test_path = config['test_path']


    Project = YOLOv10_method(train_path=train_path, val_path=val_path, test_path=test_path)

    if config['Train_YOLOv10']:
        Project.Train_YOLOv10(model_yaml_path, data_yaml_path, pre_model_name, epochs, batch, device)

    if config['crop_yolo_detection']:
        image_path = config['path'][' ']
        label_path = config['path']['label_path']
        restore_path = config['path']['restore_path']
        Project.crop_yolo_detection(image_path, label_path, restore_path)

    if config['inference']:
        weights = config['weights']
        show_labels = config['show_labels']
        show_conf = config['show_conf']
        show_boxes = config['show_boxes']

        Project.inference(weights=weights, save_txt=config['save_txt'], save_crop=config['save_crop'], show_labels=config['show_labels'], show_conf=config['show_conf'],
                          show_boxes=config['show_boxes'], conf=config['conf'], iou=config['iou'], exist_ok=config['exist_ok'])

    if config['single_image_txt_to_image']:
        single_txt_file = config['single_txt_file']
        single_image_file = config['single_image_file']
        Project.single_image_txt_to_image(single_txt_file=single_txt_file, single_image_file=single_image_file)


