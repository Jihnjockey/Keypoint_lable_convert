#将图片和标签分别分为训练集验证集和测试集
#存放格式为Coco数据集格式
#datasets-images|-train
#         |     |-val
#         |     |-test
#         -labels
import json
import os
from pathlib import Path
import numpy as np
import yaml
import random
from PIL import Image
from tqdm import tqdm
import shutil

datasets_orig_dir = r'D:\Bone_Joint_Identity\JSON2YOLO-master\generated_YOLOtxt\labels'  # 数据集原始位置，下分images和lables
imagefile_dir=r'D:\Bone_Joint_Identity\ultralytics-main\Datasets\images'
final_dir = r'D:\Bone_Joint_Identity\ultralytics-main\20_datasets'
label_format = 'txt'
image_format = 'jpg'
rate = 0.2  # val+test 占总数据集的比例

def makecocofolder(final_dir):
    final_dir=Path(final_dir)
    os.makedirs(final_dir/'images')
    os.makedirs(final_dir/'images'/'train' )
    os.makedirs(final_dir/'images'/'val')
    os.makedirs(final_dir/'images'/'test')

    os.makedirs(final_dir/'labels')
    os.makedirs(final_dir/'labels'/'train' )
    os.makedirs(final_dir/'labels'/'val')
    os.makedirs(final_dir/'labels'/'test')


def move_imgsnlabels(mode,label_list):
    mode=mode
    pbar = tqdm(label_list)
    for label in pbar:
        pbar.set_description("Processing %s" % mode)
        if label.split('.')[-1]==label_format:
            shutil.copy(os.path.join(datasets_orig_dir,label) ,os.path.join(final_dir,'labels\\'+mode)) #复制label
            imagename= label.split('.')[0]
            try:
                shutil.copy(os.path.join(imagefile_dir, imagename+'.'+image_format), os.path.join(final_dir, 'images\\'+mode)) #复制image
            except:
                print("not found "+imagename)

        else:
            print("format error!")




if __name__ == '__main__':

    final_dir=Path(final_dir)

    if not os.path.exists(final_dir):
        makecocofolder(final_dir)
    else:
        shutil.rmtree(final_dir)
        makecocofolder(final_dir)

    label_list=os.listdir(datasets_orig_dir)
    total_label_nums=len(label_list)
    valtest= random.sample(label_list, int(total_label_nums*rate)*2)
    half=int(len(valtest)/2)
    val_list=valtest[:half] #元素：e.g. 1619.json
    test_list=valtest[half:]
    train_lsit = [x for x in label_list if x not in valtest]

    dict1=    {'train':train_lsit,
               'val':val_list,
               'test':test_list}

    for key in dict1:
        move_imgsnlabels(key,dict1[key])





