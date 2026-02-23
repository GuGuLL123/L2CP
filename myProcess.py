from lib.datasetV2 import data_preprocess
import os
import cv2,imageio,PIL
import numpy as np
from lib.extract_patches import load_data, my_PreProc, is_patch_inside_FOV

def myprosess():
    # data_path_list = './prepare_dataset/data_path_list/CHASEDB1/train.txt'
    image_path = '/data/ylgu/Medical/DG/Multi-Source/VesselDatasets/XCAD_shi/test/img'
    image_list = os.listdir(image_path)
    for index in range(len(image_list)):
        img_path = os.path.join(image_path, image_list[index])
        img = PIL.Image.open(img_path)
        img = np.asarray(img)
        if len(img.shape) == 2:
            img = img[:,:,np.newaxis]
            img = np.repeat(img, 3, axis=2)
        img = img[np.newaxis, :, :,:]
        img = np.transpose(img, (0, 3, 1, 2))
        train_imgs, _ = my_PreProc(img)
        train_imgs = train_imgs[0,0, :, :]
        cv2.imwrite('/data/ylgu/Medical/DG/Multi-Source/VesselDatasets/XCAD_shi/test/processed_images/' + image_list[index], train_imgs*255)
    print('1')

def split_data():
    image_path = '/data/ylgu/Medical/DG/Multi-Source/VesselDatasets/XCAD_shi/test/img'
    image_list = os.listdir(image_path)
    max_index = int(0.8 * len(image_list))
    train_file = open('/data/ylgu/Medical/DG/Multi-Source/VesselDatasets/XCAD_shi/test/test_train.txt', 'w')
    test_file = open('/data/ylgu/Medical/DG/Multi-Source/VesselDatasets/XCAD_shi/test/test_test.txt', 'w')
    index = 0
    for image in image_list:
        if index < max_index:
            train_file.write('/data/ylgu/Medical/DG/Multi-Source/VesselDatasets/XCAD_shi/test/img/' + image + ' ' + '/data/ylgu/Medical/DG/Multi-Source/VesselDatasets/XCAD_shi/test/gt/' + image + '\n')
        else:
            test_file.write('/data/ylgu/Medical/DG/Multi-Source/VesselDatasets/XCAD_shi/test/img/' + image + ' ' + '/data/ylgu/Medical/DG/Multi-Source/VesselDatasets/XCAD_shi/test/gt/' + image + '\n')
        index += 1

myprosess()