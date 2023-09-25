#骨骼关键点为2D，无visible，json由labelme标注生成
#标注时，一个图片有多人时，设置group_id，从0开始；只有一人时无需设置，为None
import json
import os
from pathlib import Path
import numpy as np
import requests
import yaml
from PIL import Image
from tqdm import tqdm

keypointlist=['cls','cx','cy','w','h','nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder',  'left_elbow',
'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle']

savedir='./generated_YOLOtxt/labels' #保存的txt标签存放位置


def convert(file): #读取的json里骨骼关键点为2D,即无Visible,
    # Convert Labelbox JSON labels to YOLO labels
    names = []  # class names

    with open(file) as f:
        data = json.load(f)  # load JSON
    peoples= np.zeros((1,39), dtype = np.float16)
    Height = data['imageHeight']
    Width = data['imageWidth']


    keypoints_dict=data['shapes']

    for shape in keypoints_dict:


        if shape['label'] == 'person':
            if shape['group_id'] and int(shape['group_id'])> 0:
                newarray = np.zeros((1, 39), dtype=np.float16)
                peoples = np.concatenate((peoples, newarray))

            pidx=shape['group_id'] if shape['group_id'] else 0

            peoples[pidx][0]=0


            left=shape['points'][0][0] if shape['points'][0][0]>=0.0 else 0.0
            top = shape['points'][0][1] if shape['points'][0][1]  >= 0.0 else 0.0
            right = shape['points'][1][0] if shape['points'][0][0]<=Width else Width
            bottom = shape['points'][1][1] if shape['points'][0][1] <= Height else Height  #处理越界

            cx, cy = (left+right)/(2*Width ), (top+bottom)/(2*Height) #归一化
            w,h= (right-left)/Width, (bottom-top)/Height

            peoples[pidx][1] =cx
            peoples[pidx][2] =cy
            peoples[pidx][3] =w
            peoples[pidx][4] =h
        else:
            kpidx=  keypointlist.index(shape['label'])
            peoples_idx=2*kpidx-5
            peoples[pidx][peoples_idx]= shape['points'][0][0]/Width
            peoples[pidx][peoples_idx + 1] = shape['points'][0][1] /Height
    
    
    txtname=data['imagePath'].split('\\')[-1].split('.')[0]
    dstpath=os.path.join(savedir,txtname+'.txt')
    with open(dstpath, 'w') as f:
        for people in peoples:
            for i in range(len(people)):
                if i==0:
                    f.write(str(int(people[i]))+' ')
                else:
                    f.write(str(people[i])+' ')
            
            f.write('\n')
        

if __name__ == '__main__':

    originaldir=r'D:\Bone_Joint_Identity\ultralytics-main\Datasets\labels'  #要转换的json标签集
    lableslist=os.listdir(originaldir)
    lableslists=tqdm(lableslist)
    for label in lableslists:
        jsonfile=os.path.join(originaldir,label)
        convert(jsonfile)
        lableslists.set_description("Processing %s" % label)