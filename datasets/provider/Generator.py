import math
import os
import random

import pandas as pd
import numpy as np
import csv
import cv2


#row_data_dir_path = '/Users/zale/project/myself/registration_cnn_ntg/datasets/row_data/VOC/'
from tnf_transform.img_process import random_affine, generator_affine_param
from train import init_seeds


def read_row_data(data_path):
    image_name_list = os.listdir(data_path)
    #print(image_name_list)
    return image_name_list

def affine_transform(image,param):
    height = image.shape[0]
    width = image.shape[1]
    image = cv2.warpAffine(image,param,(width,height))
    return image



def generate_result_dict(row_data_dir_path,output_path,use_custom_random_aff = False):
    image_name_list = read_row_data(row_data_dir_path)
    param_list = []
    for i in range(len(image_name_list)):
        if use_custom_random_aff:
            random_param_dict = random_affine(to_dict= True)
        else:
            random_param_dict = generator_affine_param(to_dict=True)

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
init_seeds(seed= 23422)
#row_data_dir_path = '/home/zlk/datasets/coco_test2017'
row_data_dir_path = '../row_data/COCO'

use_custom_random_aff = True

if use_custom_random_aff:
    output_path = '../row_data/label_file/coco_test2017_custom_param.csv'
else:
    output_path = '../row_data/label_file/coco_test2017_paper_param.csv'

generate_result_dict(row_data_dir_path,output_path,use_custom_random_aff=use_custom_random_aff)

#print(random_affine(to_dict= True))