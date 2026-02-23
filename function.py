import random
from os.path import join

import numpy as np
import torch

from lib.extract_patches import get_data_train
from lib.losses.loss import *
from lib.visualize import group_images, save_img
from lib.common import *
from lib.dataset import TrainDataset
from torch.utils.data import DataLoader
from collections import OrderedDict
from lib.metrics import Evaluate
from lib.visualize import group_images, save_img
from lib.extract_patches import get_data_train
from lib.datasetV2 import data_preprocess,create_patch_idx,TrainDatasetV2,TrainDatasetV3
from tqdm import tqdm
import skimage.morphology
import scipy.ndimage
import wandb

def sigmoid_rampup(current_epoch,rampup_weight, rampup_length=200):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current_epoch, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return rampup_weight *float(np.exp(-5.0 * phase * phase))

def get_thickness(masks):
    thicknesses = np.zeros_like(masks, dtype=np.float32)
    for i, m in enumerate(masks):
        m = m.squeeze()
        skeleton = skimage.morphology.skeletonize(m)
        distance = scipy.ndimage.distance_transform_edt(1 - skeleton)
        contours = skimage.morphology.binary_dilation(m, np.ones((3, 3))) - m
        distance = distance * contours
        point2contour_coords = scipy.ndimage.distance_transform_edt(1 - contours, return_distances=False, return_indices=True)
        distance = distance[point2contour_coords[0], point2contour_coords[1]]
        distance = distance * m
        distance_max = np.max(distance)
        distance_min = np.min(distance)
        distance = (distance - distance_min) / (distance_max - distance_min+ 1e-8)

        thicknesses[i] = distance

    return thicknesses
# ========================get dataloader==============================
def get_dataloader(args):
    """
    该函数将数据集加载并直接提取所有训练样本图像块到内存，所以内存占用率较高，容易导致内存溢出
    """
    patches_imgs_train, patches_masks_train = get_data_train(
        data_path_list = args.train_data_path_list,
        patch_height = args.train_patch_height,
        patch_width = args.train_patch_width,
        N_patches = args.N_patches,
        inside_FOV = args.inside_FOV #select the patches only inside the FOV  (default == False)
    )
    val_ind = random.sample(range(patches_masks_train.shape[0]),int(np.floor(args.val_ratio*patches_masks_train.shape[0])))
    train_ind =  set(range(patches_masks_train.shape[0])) - set(val_ind)
    train_ind = list(train_ind)

    train_set = TrainDataset(patches_imgs_train[train_ind,...],patches_masks_train[train_ind,...],mode="train")
    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True, num_workers=6)

    val_set = TrainDataset(patches_imgs_train[val_ind,...],patches_masks_train[val_ind,...],mode="val")
    val_loader = DataLoader(val_set, batch_size=args.batch_size,
                            shuffle=False, num_workers=6)
    # Save some samples of feeding to the neural network
    if args.sample_visualization:
        N_sample = min(patches_imgs_train.shape[0], 50)
        save_img(group_images((patches_imgs_train[0:N_sample, :, :, :]*255).astype(np.uint8), 10),
                join(args.outf, args.save, "sample_input_imgs.png"))
        save_img(group_images((patches_masks_train[0:N_sample, :, :, :]*255).astype(np.uint8), 10),
                join(args.outf, args.save,"sample_input_masks.png"))
    return train_loader,val_loader




def get_dataloaderV2(args):
    """
    该函数加载数据集所有图像到内存，并创建训练样本提取位置的索引，所以占用内存量较少，
    测试结果表明，相比于上述原始的get_dataloader方法并不会降低训练效率
    """

    imgs_train, closing_weight, masks_train, fovs_train = data_preprocess(data_path_list = args.train_data_path_list)


    # print('1')


    patches_idx = create_patch_idx(fovs_train, args)

    train_idx,val_idx = np.vsplit(patches_idx, (int(np.floor((1-args.val_ratio)*patches_idx.shape[0])),))

    train_set = TrainDatasetV2(imgs_train, masks_train, fovs_train,train_idx,mode="train",args=args, closing_weight=closing_weight)
    # train_set = TrainDatasetV3(imgs_train, masks_train, fovs_train, train_idx,thicknesses, mode="train", args=args)
    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True, num_workers=0)

    val_set = TrainDatasetV2(imgs_train, masks_train, fovs_train,val_idx,mode="val",args=args)
    val_loader = DataLoader(val_set, batch_size=args.batch_size,
                            shuffle=False, num_workers=0)

    # # Save some samples of feeding to the neural network
    # if args.sample_visualization:
    #     visual_set = TrainDatasetV2(imgs_train, masks_train, fovs_train,val_idx,mode="val",args=args)
    #     # visual_set = TrainDatasetV3(imgs_train, masks_train, fovs_train, val_idx, mode="val", args=args)
    #     visual_loader = DataLoader(visual_set, batch_size=1,shuffle=True, num_workers=0)
    #     N_sample = 50
    #     visual_imgs = np.empty((N_sample,1,args.train_patch_height, args.train_patch_width))
    #     visual_masks = np.empty((N_sample,1,args.train_patch_height, args.train_patch_width))
    #     # visual_thicknesses = np.empty((N_sample,1,args.train_patch_height, args.train_patch_width))
    #     for i, (img, mask) in tqdm(enumerate(visual_loader)):
    #         visual_imgs[i] = np.squeeze(img.numpy(),axis=0)
    #         visual_masks[i,0] = np.squeeze(mask.numpy(),axis=0)
    #         # visual_thicknesses[i,0] = np.squeeze(thickness.numpy(),axis=0)
    #         if i>=N_sample-1:
    #             break
    #     save_img(group_images((visual_imgs[0:N_sample, :, :, :]*255).astype(np.uint8), 10),
    #             join(args.outf, args.save, "sample_input_imgs.png"))
    #     save_img(group_images((visual_masks[0:N_sample, :, :, :]*255).astype(np.uint8), 10),
    #             join(args.outf, args.save,"sample_input_masks.png"))
    #     # save_img(group_images((visual_thicknesses[0:N_sample, :, :, :]*255).astype(np.uint8), 10),
    #     #         join(args.outf, args.save,"sample_input_thicknesses.png"))
    return train_loader,val_loader


def get_dataloaderV2_trainwithfusion(args):
    """
    该函数加载数据集所有图像到内存，并创建训练样本提取位置的索引，所以占用内存量较少，
    测试结果表明，相比于上述原始的get_dataloader方法并不会降低训练效率
    """

    imgs_train, closing_weight, masks_train, fovs_train = data_preprocess(data_path_list = args.train_data_path_list)
    imgs_train_fusion, closing_weight_fusion, masks_train_fusion, fovs_train_fusion = data_preprocess(data_path_list = args.pasted_vessel_path_list_train_with_fusion)

    # print('1')


    patches_idx = create_patch_idx(fovs_train, args)
    patches_idx_fusion = create_patch_idx(fovs_train_fusion, args)

    train_idx,val_idx = np.vsplit(patches_idx, (int(np.floor((1-args.val_ratio)*patches_idx.shape[0])),))

    train_set = TrainDatasetV2(imgs_train, masks_train, fovs_train,train_idx,mode="train",args=args, closing_weight=closing_weight)
    train_set_fusion = TrainDatasetV2(imgs_train_fusion, masks_train_fusion, fovs_train_fusion, patches_idx_fusion, mode="train", args=args,closing_weight=closing_weight_fusion)

    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True, num_workers=16)
    train_loader_fusion = DataLoader(train_set_fusion, batch_size=args.batch_size_fusion,
                              shuffle=True, num_workers=16)

    val_set = TrainDatasetV2(imgs_train, masks_train, fovs_train,val_idx,mode="val",args=args)
    val_loader = DataLoader(val_set, batch_size=args.batch_size,
                            shuffle=False, num_workers=16)

    return train_loader, train_loader_fusion, val_loader


def get_dataloaderV3(args):
    """
    该函数加载数据集所有图像到内存，并创建训练样本提取位置的索引，所以占用内存量较少，
    测试结果表明，相比于上述原始的get_dataloader方法并不会降低训练效率
    """
    imgs_train, masks_train, fovs_train = data_preprocess(data_path_list = args.train_data_path_list)
    # print('1')

    thicknesses = get_thickness(masks_train)[:, 0, :, :]
    thicknesses = thicknesses[:, np.newaxis, :, :]



    patches_idx = create_patch_idx(fovs_train, args)

    train_idx,val_idx = np.vsplit(patches_idx, (int(np.floor((1-args.val_ratio)*patches_idx.shape[0])),))

    # train_set = TrainDatasetV2(imgs_train, masks_train, fovs_train,train_idx,mode="train",args=args)
    train_set = TrainDatasetV3(imgs_train, masks_train, fovs_train, train_idx,thicknesses, mode="train", args=args)
    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True, num_workers=0)

    val_set = TrainDatasetV2(imgs_train, masks_train, fovs_train,val_idx,mode="val",args=args)
    val_loader = DataLoader(val_set, batch_size=args.batch_size,
                            shuffle=False, num_workers=0)

    # Save some samples of feeding to the neural network
    if args.sample_visualization:
        # visual_set = TrainDatasetV2(imgs_train, masks_train, fovs_train,val_idx,mode="val",args=args)
        visual_set = TrainDatasetV3(imgs_train, masks_train, fovs_train, val_idx,thicknesses, mode="val", args=args)
        visual_loader = DataLoader(visual_set, batch_size=1,shuffle=True, num_workers=0)
        N_sample = 50
        visual_imgs = np.empty((N_sample,1,args.train_patch_height, args.train_patch_width))
        visual_masks = np.empty((N_sample,1,args.train_patch_height, args.train_patch_width))
        visual_thicknesses = np.empty((N_sample,1,args.train_patch_height, args.train_patch_width))
        for i, (img, mask,thickness) in tqdm(enumerate(visual_loader)):
            visual_imgs[i] = np.squeeze(img.numpy(),axis=0)
            visual_masks[i,0] = np.squeeze(mask.numpy(),axis=0)
            visual_thicknesses[i,0] = np.squeeze(thickness.numpy(),axis=0)
            if i>=N_sample-1:
                break
        save_img(group_images((visual_imgs[0:N_sample, :, :, :]*255).astype(np.uint8), 10),
                join(args.outf, args.save, "sample_input_imgs.png"))
        save_img(group_images((visual_masks[0:N_sample, :, :, :]*255).astype(np.uint8), 10),
                join(args.outf, args.save,"sample_input_masks.png"))
        save_img(group_images((visual_thicknesses[0:N_sample, :, :, :]*255).astype(np.uint8), 10),
                join(args.outf, args.save,"sample_input_thicknesses.png"))
    return train_loader,val_loader

# =======================train======================== 
def train(train_loader,net,ce_loss,dice_loss,optimizer,device,epoch):
    net.train()
    train_loss = AverageMeter()

    for batch_idx, (inputs, targets, closing_weight) in tqdm(enumerate(train_loader), total=len(train_loader)):

        inputs, targets = inputs.to(device), targets.to(device)
        closing_weight = closing_weight.to(device)

        optimizer.zero_grad()
        outputs = net(inputs)
        outputs_soft = torch.softmax(outputs, dim=1)
        loss1 = ce_loss(outputs, targets)
        loss2 = dice_loss(outputs_soft, targets.unsqueeze(1))

        # iter_weight = sigmoid_rampup(epoch, 1, 20)
        # loss_perpixel = ce_perpixel_loss(outputs, targets)
        # loss_closing = iter_weight * torch.mean(loss_perpixel * closing_weight)
        loss = loss1 + loss2
        loss.backward()
        # wandb.log({'loss_ce': loss1})
        # wandb.log({'loss_dice': loss2})
        optimizer.step()

        train_loss.update(loss.item(), inputs.size(0))
    log = OrderedDict([('train_loss',train_loss.avg)])
    return log

###在训练过程加入fusion数据增强的训练
def train_fusion(train_loader,train_loader_fusion,net,ce_loss,dice_loss,optimizer,device,epoch):
    net.train()
    train_loss = AverageMeter()
    loader = zip(train_loader, train_loader_fusion)
    for batch_idx, ((inputs, targets, _),(inputs_fusion, targets_fusion, _)) in tqdm(enumerate(loader), total=len(train_loader)):

        inputs, targets = inputs.to(device), targets.to(device)
        if random.random() < 0.5:
            inputs_fusion, targets_fusion = inputs_fusion.to(device), targets_fusion.to(device)
            indices = torch.randperm(inputs.size(0))[:inputs_fusion.size(0)]
            inputs[indices] = inputs_fusion
            targets[indices] = targets_fusion
        optimizer.zero_grad()
        outputs = net(inputs)
        outputs_soft = torch.softmax(outputs, dim=1)
        loss1 = ce_loss(outputs, targets)
        loss2 = dice_loss(outputs_soft, targets.unsqueeze(1))


        loss = loss1 + loss2
        loss.backward()
        # wandb.log({'loss_ce': loss1})
        # wandb.log({'loss_dice': loss2})
        optimizer.step()

        train_loss.update(loss.item(), inputs.size(0))
    log = OrderedDict([('train_loss',train_loss.avg)])
    return log

def train_thickness(train_loader,net,ce_perpixel_loss,ce_loss,dice_loss,optimizer,device,epoch):
    net.train()
    train_loss = AverageMeter()

    for batch_idx, (inputs, targets,thicknesses) in tqdm(enumerate(train_loader), total=len(train_loader)):
        inputs, targets,thicknesses = inputs.to(device), targets.to(device),thicknesses.to(device)
        thicknesses_weight = sigmoid_rampup(epoch, 5, 20)
        thicknesses[thicknesses==0] = 1.1
        thicknesses = (torch.tensor(1.1) - thicknesses)
        optimizer.zero_grad()

        outputs = net(inputs)
        outputs_soft = torch.softmax(outputs, dim=1)
        loss_perpixel = ce_perpixel_loss(outputs, targets)
        loss_thicknesses = thicknesses_weight * torch.mean(loss_perpixel * thicknesses)
        loss_ce = ce_loss(outputs, targets)
        loss_dice = dice_loss(outputs_soft, targets.unsqueeze(1))
        loss = loss_ce + loss_dice + loss_thicknesses
        loss.backward()
        optimizer.step()

        train_loss.update(loss.item(), inputs.size(0))
    log = OrderedDict([('train_loss',train_loss.avg)])
    return log

# ========================val=============================== 
def val(val_loader,net,criterion,device):
    net.eval()
    val_loss = AverageMeter()
    evaluater = Evaluate()
    with torch.no_grad():
        for batch_idx, (inputs, targets,_) in tqdm(enumerate(val_loader), total=len(val_loader)):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            val_loss.update(loss.item(), inputs.size(0))

            outputs = outputs.data.cpu().numpy()
            targets = targets.data.cpu().numpy()
            evaluater.add_batch(targets,outputs[:,1])
    log = OrderedDict([('val_loss', val_loss.avg), 
                       ('val_acc', evaluater.confusion_matrix()[1]), 
                       ('val_f1', evaluater.f1_score()),
                       ('val_auc_roc', evaluater.auc_roc())])
    return log



# #训练血管和背景融合的函数
# def train_fusion(train_loader,net,ce_loss,dice_loss,optimizer,device,epoch):
#     net.train()
#     # train_loss = AverageMeter()
#     L1LOSS = nn.L1Loss()
#     L2LOSS = nn.MSELoss()
#     for batch_idx, (inputs, targets, closing_weight) in tqdm(enumerate(train_loader), total=len(train_loader)):
#         inputs, targets = inputs.to(device), targets.to(device)
#         targets = targets.unsqueeze(1)
#         inputs_np = inputs.cpu().numpy()
#         closing_inputs = np.zeros_like(inputs_np)
#         for index in range(inputs_np.shape[0]):
#             closing_inputs[index,0,:,:] = cv2.morphologyEx(inputs_np[index,0,:,:],cv2.MORPH_CLOSE,np.ones((13,13)))
#         closing_inputs = torch.from_numpy(closing_inputs).to(device)
#
#         final_inputs = torch.cat((closing_inputs,targets),dim=1)
#         optimizer.zero_grad()
#         outputs = net(final_inputs)
#         outpus_sigmoid = torch.sigmoid(outputs)
#         l1loss = L1LOSS(outpus_sigmoid,inputs)
#         l2loss = L2LOSS(outpus_sigmoid,inputs)
#         loss = l1loss + l2loss
#
#         loss.backward()
#         wandb.log({'L1': l1loss})
#         wandb.log({'L2': l2loss})
#         optimizer.step()
#
#     return 0