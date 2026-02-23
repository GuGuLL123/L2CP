"""
This dataset is used reduce memory usage during training
"""
from torch.utils.data import Dataset
import torch
import numpy as np
import random
from imgaug import augmenters as iaa
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms.functional import normalize

from .extract_patches import load_data, my_PreProc, is_patch_inside_FOV
from .dataset import RandomCrop,RandomCropWidth, RandomFlip_LR,RandomFlip_LRWidth, RandomFlip_UD,RandomFlip_UDWidth, RandomRotate,RandomRotateWidth, Compose,ComposeWidth

class TrainDatasetV2(Dataset):
    def __init__(self, imgs,masks,fovs,patches_idx,mode,args):
        self.imgs = imgs

        self.masks = masks
        self.fovs = fovs
        self.patch_h, self.patch_w = args.train_patch_height_tta, args.train_patch_width_tta
        self.patches_idx = patches_idx
        self.inside_FOV = args.inside_FOV
        self.transforms = None
        if mode == "train":
            self.transforms = Compose([
                # RandomResize([56,72],[56,72]),
                # RandomCrop((48, 48)),
                RandomFlip_LR(prob=0.5),
                RandomFlip_UD(prob=0.5),
                RandomRotate()
            ])

    def __len__(self):
        return len(self.patches_idx)

    def __getitem__(self, idx):
        n, x_center, y_center = self.patches_idx[idx]

        data = self.imgs[n,:,y_center-int(self.patch_h/2):y_center+int(self.patch_h/2),x_center-int(self.patch_w/2):x_center+int(self.patch_w/2)]
        mask = self.masks[n,:,y_center-int(self.patch_h/2):y_center+int(self.patch_h/2),x_center-int(self.patch_w/2):x_center+int(self.patch_w/2)]

        data = torch.from_numpy(data).float()
        mask = torch.from_numpy(mask).long()

        if self.transforms:
            data, mask = self.transforms(data, mask)
        # if data.shape !=[1, 128, 128]:
        #     print(idx)
        #     print(data.shape,mask.shape)
        return data, mask.squeeze(0)


class TrainDatasetV3(Dataset):
    def __init__(self, imgs,masks,fovs,patches_idx,thicknesses,mode,args):
        self.imgs = imgs

        self.masks = masks
        self.fovs = fovs
        self.thicknesses = thicknesses
        self.patch_h, self.patch_w = args.train_patch_height, args.train_patch_width
        self.patches_idx = patches_idx
        self.inside_FOV = args.inside_FOV
        self.transforms = None
        self.use_gamma = args.use_gamma
        if mode == "train":
            self.transforms = ComposeWidth([
                # RandomResize([56,72],[56,72]),
                RandomCropWidth((48, 48)),
                RandomFlip_LRWidth(prob=0.5),
                RandomFlip_UDWidth(prob=0.5),
                RandomRotateWidth()
            ])

    def __len__(self):
        return len(self.patches_idx)

    def __getitem__(self, idx):
        n, x_center, y_center = self.patches_idx[idx]

        data = self.imgs[n,:,y_center-int(self.patch_h/2):y_center+int(self.patch_h/2),x_center-int(self.patch_w/2):x_center+int(self.patch_w/2)]
        mask = self.masks[n,:,y_center-int(self.patch_h/2):y_center+int(self.patch_h/2),x_center-int(self.patch_w/2):x_center+int(self.patch_w/2)]
        thickness = self.thicknesses[n,:,y_center-int(self.patch_h/2):y_center+int(self.patch_h/2),x_center-int(self.patch_w/2):x_center+int(self.patch_w/2)]

        if self.use_gamma:
            if random.random() > 0.5:
                contrast = iaa.GammaContrast((0.5, 2))
                data = contrast.augment_image(data)

        data = torch.from_numpy(data).float()
        mask = torch.from_numpy(mask).long()
        thickness = torch.from_numpy(thickness).float()

        if self.transforms:
            data, mask,thickness = self.transforms(data, mask,thickness)
        return data, mask.squeeze(0),thickness.squeeze(0)

#----------------------Related Methon--------------------------------------
def data_preprocess(data_path_list):
    train_imgs_original, train_masks, train_FOVs = load_data(data_path_list)
    # save_img(group_images(train_imgs_original[0:20,:,:,:],5),'imgs_train.png')#.show()  #check original train imgs 

    train_imgs = my_PreProc(train_imgs_original)
    train_masks = train_masks//255
    train_FOVs = train_FOVs//255
    return train_imgs, train_masks, train_FOVs

def create_patch_idx(img_fovs, args):
    assert len(img_fovs.shape)==4
    N,C,img_h,img_w = img_fovs.shape
    res = np.empty((args.N_patches_tta,3),dtype=int)
    print("")

    seed=2021
    count = 0
    while count < args.N_patches_tta:
        random.seed(seed) # fuxian
        seed+=1
        n = random.randint(0,N-1)
        x_center = random.randint(0+int(args.train_patch_width_tta/2),img_w-int(args.train_patch_width_tta/2))
        y_center = random.randint(0+int(args.train_patch_height_tta/2),img_h-int(args.train_patch_height_tta/2))

        #check whether the patch is contained in the FOV
        if args.inside_FOV=='center' or args.inside_FOV == 'all':
            if not is_patch_inside_FOV(x_center,y_center,img_fovs[n,0],args.train_patch_height,args.train_patch_width,mode=args.inside_FOV):
                continue
        res[count] = np.asarray([n,x_center,y_center])
        count+=1

    return res

