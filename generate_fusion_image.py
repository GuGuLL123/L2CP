import joblib, copy
import numpy as np
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch, sys
from tqdm import tqdm

from collections import OrderedDict
from lib.visualize import save_img, group_images, concat_result
import os
import argparse
from lib.logger import Logger, Print_Logger
from lib.extract_patches import *
from os.path import join
from lib.dataset import TestDataset,TestDataset_TTAoneImage
from lib.metrics import Evaluate
import models
import torch.nn as nn
from lib.common import setpu_seed, dict_round
from config import parse_args
from lib.pre_processing import my_PreProc
import torch.optim as optim
import fusion_tta
import tent
import cotta
import cv2
setpu_seed(2021)

def paste_vessel(closing_img,closing_img_FOV, vessel_img,vessel_gt):
    generate_veseel = np.zeros_like(closing_img).astype(np.float32)
    generate_veseel[vessel_gt > 0] = vessel_img[vessel_gt > 0]
    generate_veseel = generate_veseel / np.max(generate_veseel) * 30

    closing_img[vessel_gt > 0] = closing_img[vessel_gt > 0] - generate_veseel[vessel_gt > 0]
    closing_img[closing_img < 0] = 0
    closing_img[closing_img_FOV == 0] = 0
    smoothed_foreground = cv2.GaussianBlur(closing_img, (3, 3), 0)
    smoothed_image = closing_img * (1 - vessel_gt) + smoothed_foreground * vessel_gt
    back_closing = smoothed_image


    vessel_gt_copy = np.copy(vessel_gt)
    vessel_gt_copy[closing_img_FOV == 0] = 0
    return back_closing,vessel_gt_copy

def paste_vessel_local_contrast(closing_img,closing_img_FOV, vessel_img,vessel_gt):
    ###计算源域血管local差值
    w, h = vessel_img.shape
    fore_contrast_map = np.zeros_like(vessel_img).astype(np.float32)
    local_kernel_size = 7
    for i in range(w):
        for j in range(h):
            if vessel_gt[i, j] > 0:
                fore_mean = 0
                fore_num = 0
                i_min = max(0, i - local_kernel_size)
                i_max = min(h, i + local_kernel_size)
                j_min = max(0, j - local_kernel_size)
                j_max = min(w, j + local_kernel_size)
                for ii in range(i_min, i_max):
                    for jj in range(j_min, j_max):
                        if vessel_gt[ii, jj] == 0:
                            fore_mean = fore_mean + vessel_img[ii, jj]
                            fore_num += 1
                if fore_num > 0:
                    fore_mean = fore_mean / fore_num
                    fore_contrast_map[i, j] = fore_mean - vessel_img[i, j]
    fore_contrast_map[fore_contrast_map < 0 ] = 0
    # 进行目标域血管local 背景均值计算
    back_mean_map = np.zeros_like(closing_img).astype(np.float32)
    final_fusion = np.copy(closing_img)
    for i in range(w):
        for j in range(h):
            if vessel_gt[i, j] > 0:
                fore_mean = 0
                fore_num = 0
                i_min = max(0, i - local_kernel_size)
                i_max = min(h, i + local_kernel_size)
                j_min = max(0, j - local_kernel_size)
                j_max = min(w, j + local_kernel_size)
                for ii in range(i_min, i_max):
                    for jj in range(j_min, j_max):
                        if vessel_gt[ii, jj] == 0:
                            fore_mean = fore_mean + closing_img[ii, jj]
                            fore_num += 1
                if fore_num > 0:
                    fore_mean = fore_mean / fore_num
                    back_mean_map[i, j] = fore_mean
                    final_fusion[i, j] = fore_mean - fore_contrast_map[i, j]
    final_fusion[closing_img_FOV == 0] = 0
    final_fusion[final_fusion < 0] = 0
    vessel_gt_copy = np.copy(vessel_gt)
    vessel_gt_copy[closing_img_FOV == 0] = 0

    return final_fusion,vessel_gt_copy

def test_using_target_gt(target_img_closing,target_img,target_img_FOV,target_img_gt, vessel_img,vessel_gt):
    ###计算源域血管local差值
    # cv2.imwrite('/data/ylgu/view/aug/target_img.png',target_img)
    # cv2.imwrite('/data/ylgu/view/aug/target_gt.png', target_img_gt * 255)
    # closing_img = cv2.inpaint(target_img, target_img_gt*255, 3, cv2.INPAINT_NS)
    kernel_dia = np.ones((3, 3), np.uint8)
    target_img_gt = cv2.dilate(target_img_gt, kernel_dia, iterations=1)



    w, h = vessel_img.shape
    fore_contrast_map = np.zeros_like(vessel_img).astype(np.float32)
    back_inpainting_map = target_img.copy()
    local_kernel_size = 7

    for i in range(w):
        for j in range(h):
            if target_img_gt[i, j] > 0:
                back_mean = 0
                back_num = 0
                i_min = max(0, i - local_kernel_size)
                i_max = min(h, i + local_kernel_size)
                j_min = max(0, j - local_kernel_size)
                j_max = min(w, j + local_kernel_size)
                for ii in range(i_min, i_max):
                    for jj in range(j_min, j_max):
                        if target_img_gt[ii, jj] == 0 and target_img_FOV[ii, jj] > 0:
                            back_mean = back_mean + target_img[ii, jj]
                            back_num += 1
                if back_num > 0:
                    back_mean = back_mean / back_num
                    back_inpainting_map[i, j] = back_mean

    # target_img[target_img_gt >0] = target_img_closing[target_img_gt >0]/255
    closing_img = back_inpainting_map
    # cv2.imwrite('/data/ylgu/view/aug/target_img_closing.png', target_img_closing)
    # cv2.imwrite('/data/ylgu/view/aug/target_img_new.png', closing_img)
    for i in range(w):
        for j in range(h):
            if vessel_gt[i, j] > 0:
                fore_mean = 0
                fore_num = 0
                i_min = max(0, i - local_kernel_size)
                i_max = min(h, i + local_kernel_size)
                j_min = max(0, j - local_kernel_size)
                j_max = min(w, j + local_kernel_size)
                for ii in range(i_min, i_max):
                    for jj in range(j_min, j_max):
                        if vessel_gt[ii, jj] == 0:
                            fore_mean = fore_mean + vessel_img[ii, jj]
                            fore_num += 1
                if fore_num > 0:
                    fore_mean = fore_mean / fore_num
                    fore_contrast_map[i, j] = fore_mean - vessel_img[i, j]
    fore_contrast_map[fore_contrast_map < 0 ] = 0
    # 进行目标域血管local 背景均值计算
    back_mean_map = np.zeros_like(closing_img).astype(np.float32)
    final_fusion = np.copy(closing_img)
    for i in range(w):
        for j in range(h):
            if vessel_gt[i, j] > 0:
                fore_mean = 0
                fore_num = 0
                i_min = max(0, i - local_kernel_size)
                i_max = min(h, i + local_kernel_size)
                j_min = max(0, j - local_kernel_size)
                j_max = min(w, j + local_kernel_size)
                for ii in range(i_min, i_max):
                    for jj in range(j_min, j_max):
                        if vessel_gt[ii, jj] == 0:
                            fore_mean = fore_mean + closing_img[ii, jj]
                            fore_num += 1
                if fore_num > 0:
                    fore_mean = fore_mean / fore_num
                    back_mean_map[i, j] = fore_mean
                    final_fusion[i, j] = fore_mean - fore_contrast_map[i, j]
    final_fusion[target_img_FOV == 0] = 0
    final_fusion[final_fusion < 0] = 0
    vessel_gt_copy = np.copy(vessel_gt)
    vessel_gt_copy[target_img_FOV == 0] = 0

    return final_fusion,vessel_gt_copy

def paste_vessel_mixup(closing_img,closing_img_FOV, vessel_img,vessel_gt):
    ###计算源域血管local差值
    final_fusion = closing_img*0.5 + vessel_img*0.5
    final_fusion[closing_img_FOV == 0] = 0
    vessel_gt_copy = np.copy(vessel_gt)
    vessel_gt_copy[closing_img_FOV == 0] = 0

    return final_fusion,vessel_gt_copy

def low_freq_mutate_np( amp_src, amp_trg, L=0.1 ):

    a_src = np.fft.fftshift( amp_src, axes=(0 , 1) )
    a_trg = np.fft.fftshift( amp_trg, axes=(0, 1) )

    h, w = a_src.shape
    b = (  np.floor(np.amin((h,w))*L)  ).astype(int)
    c_h = np.floor(h/2.0).astype(int)
    c_w = np.floor(w/2.0).astype(int)

    h1 = c_h-b
    h2 = c_h+b+1
    w1 = c_w-b
    w2 = c_w+b+1

    a_src[h1:h2, w1:w2] = a_trg[ h1:h2, w1:w2]
    a_src = np.fft.ifftshift( a_src, axes=(0, 1) )
    return a_src
def paste_vessel_fourier(closing_img,closing_img_FOV, vessel_img,vessel_gt):
    closing_img = closing_img.astype(np.float32)
    vessel_img = vessel_img.astype(np.float32)
    w, h = vessel_img.shape
    fore_contrast_map = np.zeros_like(vessel_img).astype(np.float32)
    ###计算源域血管local差值
    back_img = np.fft.fft2( closing_img, axes=(0, 1) )  #todo
    vessel_img = np.fft.fft2( vessel_img, axes=(0, 1) )  #todo

    # extract amplitude and phase of both ffts
    amp_back, pha_back = np.abs(back_img), np.angle(back_img)
    amp_vessel, pha_vessel = np.abs(vessel_img), np.angle(vessel_img)

    # mutate the amplitude part of source with target
    amp_src_ = low_freq_mutate_np( amp_vessel, amp_back, L=0.5 )

    # mutated fft of source
    fft_src_ = amp_src_ * np.exp( 1j * pha_vessel )

    # get the mutated image
    src_in_trg = np.fft.ifft2( fft_src_, axes=(0, 1) ) #todo
    src_in_trg = np.real(src_in_trg)
    vessel_gt_copy = np.copy(vessel_gt)
    vessel_gt_copy[closing_img_FOV == 0] = 0
    src_in_trg[src_in_trg < 0] = 0
    src_in_trg[src_in_trg>254] = 254
    return src_in_trg,vessel_gt_copy

def paste_vessel_no(closing_img,closing_img_FOV, vessel_img,vessel_gt):
    ###计算源域血管local差值
    final_fusion = np.copy(vessel_img)
    vessel_gt_copy = np.copy(vessel_gt)
    return final_fusion,vessel_gt_copy

def paste_vessel_copy_paste(closing_img,closing_img_FOV, vessel_img,vessel_gt):
    ###计算源域血管local差值
    final_fusion = np.copy(closing_img)
    final_fusion[vessel_gt > 0] = vessel_img[vessel_gt > 0]
    final_fusion[closing_img_FOV == 0] = 0
    vessel_gt_copy = np.copy(vessel_gt)
    vessel_gt_copy[closing_img_FOV == 0] = 0

    return final_fusion,vessel_gt_copy

def setup_optimizer(params):
    """Set up optimizer for tent adaptation.

    Tent needs an optimizer for test-time entropy minimization.
    In principle, tent could make use of any gradient optimizer.
    In practice, we advise choosing Adam or SGD+momentum.
    For optimization settings, we advise to use the settings from the end of
    trainig, if known, or start with a low learning rate (like 0.001) if not.

    For best results, try tuning the learning rate and batch size.
    """
    optim_method = 'Adam'
    if optim_method == 'Adam':
        return optim.Adam(params,
                    lr=1e-4,
                    betas=(0.9, 0.999),
                    weight_decay=0.0)
    elif optim_method == 'SGD':
        return optim.SGD(params,
                   lr=1e-3,
                   momentum=0.9,
                   dampening=0.0,
                   weight_decay=0.0,
                   nesterov=True)
    else:
        raise NotImplementedError

class Test_TTAoneImage():
    def __init__(self, args):
        self.args = args
        assert (args.stride_height <= args.test_patch_height and args.stride_width <= args.test_patch_width)
        # save path
        pasted_vessel_dataset_name = args.pasted_vessel_path_list.split('/')[-2]
        test_data_name = args.test_data_path_list.split('/')[-2]
        self.path_experiment = join(args.outf, args.save)+'/'+args.model_name+'/'+pasted_vessel_dataset_name+ '_' + test_data_name + '_' + self.args.tta_method + '/'

        self.test_imgs_original, self.test_imgs, self.test_masks, self.test_FOVs, self.pasted_vessel_imgs_original, self.pasted_vessel_processed ,self.pasted_vessel_gts,self.pasted_vessel_FOVs= get_data_test_overlap_tta_oneimage(
            test_data_path_list=args.test_data_path_list,
            pasted_vessel_path_list=args.pasted_vessel_path_list,
            patch_height=args.test_patch_height,
            patch_width=args.test_patch_width,
            stride_height=args.stride_height,
            stride_width=args.stride_width
        )
        self.img_height = self.test_imgs.shape[2]
        self.img_width = self.test_imgs.shape[3]

        test_set = TestDataset_TTAoneImage(self.test_imgs_original, self.test_imgs, self.test_masks, self.test_FOVs)
        self.test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=8)

    # Inference prediction process

    def tta_pasted_vessel(self):

        paseted_index_bach = [0]

        fusion_list = ['L2CP','Fourier','Mixup','CopyPaste']
        test_dataset = self.args.test_data_path_list.split('/')[-2]
        pasted_dataset = self.args.pasted_vessel_path_list.split('/')[-2]
        data_save_name = pasted_dataset+'_' + str(paseted_index_bach[0]) + '_' + test_dataset

        target_image_num = 0
        for batch_idx, (test_imgs_original,test_imgs, test_masks,test_FOVs) in tqdm(enumerate(self.test_loader), total=len(self.test_loader)):

            kernel = np.ones((13, 13), np.uint8)
            test_imgs_original_closing = np.array(test_imgs_original)[0,0]
            test_imgs_original_for_test_bound = test_imgs_original_closing.copy()
            test_masks_for_test_bound = np.array(test_masks)[0].astype(np.uint8)
            test_imgs_original_ablation = test_imgs_original_closing.copy()
            # cv2.imwrite('/data/ylgu/Medical/DG/Multi-Source/VesselDatasets_chen/CHASEDB1/fake_images/view/img/test_imgs.png',test_imgs_original_closing)
            test_imgs_original_closing = cv2.morphologyEx(test_imgs_original_closing, cv2.MORPH_CLOSE, kernel)
            # cv2.imwrite('/data/ylgu/Medical/DG/Multi-Source/VesselDatasets_chen/CHASEDB1/fake_images/view/img/test_imgs_original_closing.png',test_imgs_original_closing)
            test_imgs_closing = np.array(test_imgs)[0,0]
            if not os.path.exists('/data/ylgu/Medical/DG/Multi-Source/VesselDatasets_chen/rebuttal_miccai_view/fusion_view/{}/{}/'.format(data_save_name,test_dataset)):
                os.makedirs('/data/ylgu/Medical/DG/Multi-Source/VesselDatasets_chen/rebuttal_miccai_view/fusion_view/{}/{}'.format(data_save_name,test_dataset))
            cv2.imwrite('/data/ylgu/Medical/DG/Multi-Source/VesselDatasets_chen/rebuttal_miccai_view/fusion_view/{}/{}/{}.png'.format(data_save_name,test_dataset,str(target_image_num)), test_imgs_closing*255)

            test_imgs_closing = cv2.morphologyEx(test_imgs_closing, cv2.MORPH_CLOSE, kernel)
            test_FOVs = np.array(test_FOVs)[0]



            pasted_images_original_processed_bach = torch.zeros((len(paseted_index_bach),1,test_imgs_original_closing.shape[0],test_imgs_original_closing.shape[1])).float().cuda()
            pasted_vessel_gt_now_processed_bach = torch.zeros((len(paseted_index_bach),  test_imgs_original_closing.shape[0], test_imgs_original_closing.shape[1])).cuda()
            # for paseted_index in range(self.pasted_vessel_imgs_original.shape[0]):
            for paseted_index in paseted_index_bach:
                # if paseted_index not in [0]:
                #     continue
                pasted_vessel_img_original_now = self.pasted_vessel_imgs_original[paseted_index,0]
                pasted_vessel_processed_now = self.pasted_vessel_processed[paseted_index,0]
                pasted_vessel_gt_now = self.pasted_vessel_gts[paseted_index,0]
                # pasted_images_original,pasted_vessel_gt_now_processed = paste_vessel_no(test_imgs_original_closing, test_FOVs, pasted_vessel_img_original_now, pasted_vessel_gt_now)  #todo
                for fusion_type in fusion_list:
                    if fusion_type == 'L2CP':
                        pasted_images_original,pasted_vessel_gt_now_processed = paste_vessel_local_contrast(test_imgs_original_closing, test_FOVs, pasted_vessel_img_original_now, pasted_vessel_gt_now)   #原来实验结果
                    elif fusion_type == 'Fourier':
                        pasted_images_original, pasted_vessel_gt_now_processed = paste_vessel_fourier(test_imgs_original_ablation, test_FOVs, pasted_vessel_img_original_now, pasted_vessel_gt_now)
                    elif fusion_type == 'Mixup':
                        pasted_images_original, pasted_vessel_gt_now_processed = paste_vessel_mixup(test_imgs_original_ablation, test_FOVs, pasted_vessel_img_original_now, pasted_vessel_gt_now)
                    elif fusion_type == 'CopyPaste':
                        pasted_images_original, pasted_vessel_gt_now_processed = paste_vessel_copy_paste(test_imgs_original_ablation, test_FOVs, pasted_vessel_img_original_now, pasted_vessel_gt_now)




                # cv2.imwrite('/data/ylgu/Medical/DG/Multi-Source/VesselDatasets_chen/CHASEDB1/fake_images/view/img/pasted_images_original.png', pasted_images_original)
                # cv2.imwrite('/data/ylgu/Medical/DG/Multi-Source/VesselDatasets_chen/CHASEDB1/fake_images/view/img/pasted_vessel_gt_now.png', pasted_vessel_gt_now*255)
                    pasted_images_original = np.expand_dims(pasted_images_original, axis=(0, 1))
                    pasted_images_original = np.repeat(pasted_images_original, repeats=3, axis=1)
                    pasted_images_original_processed, _ = my_PreProc(pasted_images_original)
                    pasted_images_original_processed = pasted_images_original_processed[0,0]


                    if not os.path.exists('/data/ylgu/Medical/DG/Multi-Source/VesselDatasets_chen/rebuttal_miccai_view/fusion_view/{}/{}'.format(data_save_name,fusion_type)):
                        os.makedirs('/data/ylgu/Medical/DG/Multi-Source/VesselDatasets_chen/rebuttal_miccai_view/fusion_view/{}/{}'.format(data_save_name,fusion_type))
                    cv2.imwrite('/data/ylgu/Medical/DG/Multi-Source/VesselDatasets_chen/rebuttal_miccai_view/fusion_view/{}/{}/{}.png'.format(data_save_name,fusion_type,str(target_image_num)), pasted_images_original_processed*255)
            target_image_num += 1
                # pasted_images = paste_vessel(test_imgs_closing, test_FOVs, pasted_vessel_processed_now, pasted_vessel_gt_now)  #todo
                # pasted_images = paste_vessel_local_contrast(test_imgs_closing, test_FOVs, pasted_vessel_processed_now, pasted_vessel_gt_now)
                # pasted_images_original_processed = torch.from_numpy(pasted_images_original_processed).float().cuda()
                # # pasted_images = torch.from_numpy(pasted_images).unsqueeze(0).unsqueeze(0).cuda()
                # pasted_vessel_gt_now_processed = torch.from_numpy(pasted_vessel_gt_now_processed).unsqueeze(0).cuda()
                # pasted_images_original_processed_bach[batch_index] = pasted_images_original_processed
                # pasted_vessel_gt_now_processed_bach[batch_index] = pasted_vessel_gt_now_processed
                # print(torch.max(pasted_vessel_gt_now_processed))
    

if __name__ == '__main__':
    args = parse_args()

    ####one image
    # eval = Test_TTAoneImage(args)
    #### batch
    eval = Test_TTAoneImage(args)
    eval.tta_pasted_vessel_rgb()
    print('+'*50)
    # print(eval.evaluate())
    print('+'*50)
    eval.save_segmentation_result()
