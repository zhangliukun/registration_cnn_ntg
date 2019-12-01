import math
import os
import random

import pandas as pd
import numpy as np
import csv
import cv2


#row_data_dir_path = '/Users/zale/project/myself/registration_cnn_ntg/datasets/row_data/VOC/'
from tnf_transform.img_process import random_affine


def read_row_data(data_path):
    image_name_list = os.listdir(data_path)
    #print(image_name_list)
    return image_name_list

def affine_transform(image,param):
    height = image.shape[0]
    width = image.shape[1]
    image = cv2.warpAffine(image,param,(width,height))
    return image

#def generator_affine_param(random_t=0.2,random_s=0.2,random_alpha = 1/8,random_tps=0.4,to_dict = False):
def generator_affine_param(random_t=0.2,random_s=0.2,random_alpha = 1/8,random_tps=0.4,to_dict = False):
    alpha = (np.random.rand(1)-0.5) * 2 * np.pi * random_alpha
    theta = np.random.rand(6)

    theta[[2, 5]] = (theta[[2, 5]] - 0.5) * 2 * random_t
    theta[0] = (1 + (theta[0] - 0.5) * 2 * random_s) * np.cos(alpha)
    theta[1] = (1 + (theta[1] - 0.5) * 2 * random_s) * (-np.sin(alpha))
    theta[3] = (1 + (theta[3] - 0.5) * 2 * random_s) * np.sin(alpha)
    theta[4] = (1 + (theta[4] - 0.5) * 2 * random_s) * np.cos(alpha)
    theta = theta.reshape(2, 3)

    if to_dict:
        temp = theta.reshape(6)
        theta = {}
        theta['p0'] = temp[0]
        theta['p1'] = temp[1]
        theta['p2'] = temp[2]
        theta['p3'] = temp[3]
        theta['p4'] = temp[4]
        theta['p5'] = temp[5]

    return theta

def random_affine(degrees=20,translate=.2,scale=.2,shear=10,to_dict = False):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # https://medium.com/uruvideo/dataset-augmentation-with-random-homographies-a8f4b44830d4

    # 旋转和缩放
    R = np.eye(3)
    a = random.uniform(-degrees,degrees)
    s = random.uniform(1 - scale, 1 + scale)
    #R[:2] = cv2.getRotationMatrix2D(angle=a,center=(img.shape[1] / 2, img.shape[0] / 2),scale=s)
    R[:2] = cv2.getRotationMatrix2D(angle=a,center=(0,0),scale=s)

    # 平移
    T = np.eye(3)
    T[0, 2] = random.uniform(-translate, translate)   # x translation (rate)
    T[1, 2] = random.uniform(-translate, translate)   # y translation (rate)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    theta = S @ T @ R # Combined rotation matrix. ORDER IS IMPORTANT HERE!!

    if to_dict:
        temp = theta[0:2].reshape(6)
        theta = {}
        theta['p0'] = temp[0]
        theta['p1'] = temp[1]
        theta['p2'] = temp[2]
        theta['p3'] = temp[3]
        theta['p4'] = temp[4]
        theta['p5'] = temp[5]

    return theta

def generate_result_dict(row_data_dir_path,output_path):
    image_name_list = read_row_data(row_data_dir_path)
    param_list = []
    for i in range(len(image_name_list)):
        #random_param_dict = generator_affine_param(to_dict=True)
        random_param_dict = random_affine(to_dict= True)
        random_param_dict['image'] = image_name_list[i]
        param_list.append(random_param_dict)
        if i % 5000 == 0:
            print('第',i,"张")
        #print(image_name_list[i],param_list[i])

    write_csv(output_path,param_list)

def write_csv(output_path,datadicts):
    with open(output_path,mode='w') as csv_file:
        # 使用这个的话就直接write_row
        #employee_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        fieldnames = ['image','p0','p1','p2','p3','p4','p5']
        writer = csv.DictWriter(csv_file,fieldnames=fieldnames)

        writer.writeheader()

        # datadict = {'name':'hell','number': 'Accounting', 'age': 'November'}
        # writer.writerow(datadict)
        writer.writerows(datadicts)

def test_affine_image():
    image_name_list = read_row_data(row_data_dir_path)
    for i in range(len(image_name_list)):

        img_row = cv2.imread(row_data_dir_path+image_name_list[i])
        random_param = generator_affine_param()
        img_aff = affine_transform(img_row,random_param)

        result = np.hstack([img_row,img_aff])

        cv2.imshow('compare',result)
        cv2.waitKey(0)

#test_affine_image()
#row_data_dir_path = '/home/zlk/datasets/coco_test2017'
row_data_dir_path = '../row_data/COCO'
output_path = '../row_data/label_file/aff_param_coco_random_bigger.csv'

generate_result_dict(row_data_dir_path,output_path)

#print(random_affine(to_dict= True))