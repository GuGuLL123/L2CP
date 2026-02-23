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

from skimage.morphology import skeletonize, skeletonize_3d
setpu_seed(2021)
def cl_score(v, s):
    return np.sum(v*s)/np.sum(s)


def clDice(v_p, v_l):
    if len(v_p.shape)==2:
        tprec = cl_score(v_p,skeletonize(v_l))
        tsens = cl_score(v_l,skeletonize(v_p))
    elif len(v_p.shape)==3:
        tprec = cl_score(v_p,skeletonize_3d(v_l))
        tsens = cl_score(v_l,skeletonize_3d(v_p))
    return 2*tprec*tsens/(tprec+tsens)


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
    closing_img = closing_img.astype(np.float32)
    vessel_img = vessel_img.astype(np.float32)
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

    closing_img = back_inpainting_map
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










def setup_optimizer(params):
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

    def tta_pasted_vessel(self, net):

        kernel = np.ones((13, 13), np.uint8)
        preds = []

        net_copy = copy.deepcopy(net)
        test_frequece = 0
        for batch_idx, (test_imgs_original,test_imgs, test_masks,test_FOVs) in tqdm(enumerate(self.test_loader), total=len(self.test_loader)):
            test_frequece += 1

            if test_frequece >= 8:
                net_copy = copy.deepcopy(net)
                test_frequece = 0
            net_copy.train()
            net_copy.requires_grad_(True)
            params = net_copy.parameters()
            optimizer = setup_optimizer(params)
            print('*' * 50)
            print(self.args.tta_method)
            print('*' * 50)
            if self.args.tta_method == 'fusion':
                fusion_tta_model = fusion_tta.FusionTTA(net_copy, optimizer,
                                                        steps=1,
                                                        episodic=False)

            fusion_tta_model.reset()
            test_imgs_original_closing = np.array(test_imgs_original)[0,0]
            test_imgs_original_for_test_bound = test_imgs_original_closing.copy()
            test_masks_for_test_bound = np.array(test_masks)[0].astype(np.uint8)
            test_imgs_original_ablation = test_imgs_original_closing.copy()
            test_imgs_original_closing = cv2.morphologyEx(test_imgs_original_closing, cv2.MORPH_CLOSE, kernel)
            test_imgs_closing = np.array(test_imgs)[0,0]

            test_imgs_closing = cv2.morphologyEx(test_imgs_closing, cv2.MORPH_CLOSE, kernel)
            test_FOVs = np.array(test_FOVs)[0]

            paseted_index_bach = [0]
            batch_index = 0
            pasted_images_original_processed_bach = torch.zeros((len(paseted_index_bach),1,test_imgs_original_closing.shape[0],test_imgs_original_closing.shape[1])).float().cuda()
            pasted_vessel_gt_now_processed_bach = torch.zeros((len(paseted_index_bach),  test_imgs_original_closing.shape[0], test_imgs_original_closing.shape[1])).cuda()
            for paseted_index in paseted_index_bach:
                pasted_vessel_img_original_now = self.pasted_vessel_imgs_original[paseted_index,0]
                pasted_vessel_processed_now = self.pasted_vessel_processed[paseted_index,0]
                pasted_vessel_gt_now = self.pasted_vessel_gts[paseted_index,0]
                pasted_images_original,pasted_vessel_gt_now_processed = paste_vessel_local_contrast(test_imgs_original_closing, test_FOVs, pasted_vessel_img_original_now, pasted_vessel_gt_now)   #原来实验结果


                pasted_images_original = np.expand_dims(pasted_images_original, axis=(0, 1))
                pasted_images_original = np.repeat(pasted_images_original, repeats=3, axis=1)
                pasted_images_original_processed,_ = my_PreProc(pasted_images_original)

                pasted_images_original_processed = torch.from_numpy(pasted_images_original_processed).float().cuda()
                pasted_vessel_gt_now_processed = torch.from_numpy(pasted_vessel_gt_now_processed).unsqueeze(0).cuda()
                pasted_images_original_processed_bach[batch_index] = pasted_images_original_processed
                pasted_vessel_gt_now_processed_bach[batch_index] = pasted_vessel_gt_now_processed
                batch_index += 1
            if self.args.tta_method == 'fusion':
                outputs = fusion_tta_model(pasted_images_original_processed_bach,pasted_vessel_gt_now_processed_bach.long())
            model = fusion_tta_model.model
            model.eval()


            with torch.no_grad():
                test_imgs = test_imgs.cuda()
                outputs = model(test_imgs)
                outputs = outputs[:, 1].data.cpu().numpy()
                preds.append(outputs)
            test_masks = test_masks.numpy()[np.newaxis,...]
            test_FOVs = test_FOVs[np.newaxis,np.newaxis,...]
            y_scores, y_true = pred_only_in_FOV(outputs[np.newaxis,...], test_masks, test_FOVs)
            eval = Evaluate(save_path=self.path_experiment)
            eval.add_batch(y_true, y_scores)
            log = eval.save_all_result(plot_curve=True, save_name="performance.txt")
            print(dict_round(log, 6))
        predictions = np.concatenate(preds, axis=0)
        self.pred_patches = np.expand_dims(predictions,axis=1)

    # Evaluate ate and visualize the predicted images
    def evaluate(self):
        self.pred_imgs = self.pred_patches
        y_scores, y_true = pred_only_in_FOV(self.pred_imgs, self.test_masks, self.test_FOVs)
        eval = Evaluate(save_path=self.path_experiment)
        eval.add_batch(y_true, y_scores)
        log = eval.save_all_result(plot_curve=True, save_name="performance.txt")
        np.save('{}/result.npy'.format(self.path_experiment), np.asarray([y_true, y_scores]))
        return dict_round(log, 6)
    # Evaluate ate and visualize the predicted images

    # save segmentation imgs
    def save_segmentation_result(self):
        img_path_list, _, _ = load_file_path_txt(self.args.test_data_path_list)
        img_name_list = [item.split('/')[-1].split('.')[0] for item in img_path_list]

        kill_border(self.pred_imgs, self.test_FOVs)
        self.save_img_path = join(self.path_experiment, 'result_img')
        if not os.path.exists(join(self.save_img_path)):
            os.makedirs(self.save_img_path)
        for i in range(self.test_imgs.shape[0]):
            save_path_new = join(self.save_img_path, img_name_list[i])
            total_img = concat_result(self.test_imgs[i], self.pred_imgs[i], self.test_masks[i],save_path_new)
            save_img(total_img, join(self.save_img_path, "Result_" + img_name_list[i] + '.png'))

    # Val on the test set at each epoch
    def val(self):
        self.pred_imgs = recompone_overlap(
            self.pred_patches, self.new_height, self.new_width, self.args.stride_height, self.args.stride_width)
        self.pred_imgs = self.pred_imgs[:, :, 0:self.img_height, 0:self.img_width]

        y_scores, y_true = pred_only_in_FOV(self.pred_imgs, self.test_masks, self.test_FOVs)
        eval = Evaluate(save_path=self.path_experiment)
        eval.add_batch(y_true, y_scores)
        confusion, accuracy, specificity, sensitivity, precision = eval.confusion_matrix()
        log = OrderedDict([('val_auc_roc', eval.auc_roc()),
                           ('val_f1', eval.f1_score()),
                           ('val_acc', accuracy),
                           ('SE', sensitivity),
                           ('SP', specificity)])
        return dict_round(log, 6)



if __name__ == '__main__':
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    save_path = './experiments/segmentation/medical/VesselSeg-Pytorch/experiments/Laddernet_vessel_seg_HRF/laddernet/max'
    sys.stdout = Print_Logger(os.path.join(save_path, 'test_log.txt'))
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    if args.model_name == 'unet':
        net = models.U_Net(img_ch=args.in_channels, output_ch=args.classes).to(device)
        print('unet')
    elif args.model_name == 'laddernet':
        net = models.LadderNet(inplanes=args.in_channels, num_classes=args.classes, layers=3, filters=16).to(device)
        print('laddernet')
    cudnn.benchmark = True

    print('==> Loading checkpoint...')
    checkpoint = torch.load(join(save_path, 'best_model.pth'))
    net.load_state_dict(checkpoint['net'])

    eval = Test_TTAoneImage(args)
    eval.tta_pasted_vessel(net)
    print('+'*50)
    # eval.evaluate_cldice()
    print(eval.evaluate())
    print('+'*50)
    eval.save_segmentation_result()
