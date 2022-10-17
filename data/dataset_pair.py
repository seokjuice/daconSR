import random
import numpy as np
import torch.utils.data as data
import utils.utils_image as util
import os


class DatasetPair(data.Dataset):

    def __init__(self, opt):
        super(DatasetPair, self).__init__()
        self.opt = opt
        self.n_channels = opt['n_channels'] if opt['n_channels'] else 3
        self.sf = opt['scale'] if opt['scale'] else 4
        self.shuffle_prob = opt['shuffle_prob'] if opt['shuffle_prob'] else 0.1
        self.lq_patchsize = self.opt['lq_patchsize'] if self.opt['lq_patchsize'] else 64
        self.patch_size = self.opt['H_size'] if self.opt['H_size'] else self.lq_patchsize*self.sf
        
        self.imageList = os.listdir(opt['dataroot_H'])
        self.dataroot_H = opt['dataroot_H']
        self.dataroot_L = opt['dataroot_L']

        assert self.imageList, 'Error: H path is empty.'

    def __getitem__(self, index):

        # ------------------------------------
        # get H image
        # ------------------------------------
        H_path = self.dataroot_H + self.imageList[index]
        img_H = util.imread_uint(H_path, self.n_channels)

        L_path = self.dataroot_L + self.imageList[index]
        img_L = util.imread_uint(L_path, self.n_channels)

        img_name, ext = os.path.splitext(os.path.basename(H_path))
        H, W, C = img_H.shape

        # ------------------------------------
        # if train, get L/H patch pair
        # ------------------------------------
        if self.opt['phase'] == 'train':

            H, W, C = img_H.shape

            mode = random.randint(0, 7)
            img_H,img_L = util.augment_img(img_H,img_L, mode=mode)

            img_H = util.uint2single(img_H)
            img_L = util.uint2single(img_L)

        else:
            img_H = util.uint2single(img_H)
            img_L = util.uint2single(img_L)

        # ------------------------------------
        # L/H pairs, HWC to CHW, numpy to tensor
        # ------------------------------------
        img_H, img_L = util.single2tensor3(img_H), util.single2tensor3(img_L)

        if L_path is None:
            L_path = H_path

        return {'L': img_L, 'H': img_H, 'L_path': L_path, 'H_path': H_path}

    def __len__(self):
        return len(self.imageList)
