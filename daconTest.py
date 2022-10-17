import os.path
import math
import argparse
import time
import random
import numpy as np
from collections import OrderedDict
import logging
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch

from utils import utils_logger
from utils import utils_image as util
from utils import utils_option as option
from utils.utils_dist import get_dist_info, init_dist

from data.select_dataset import define_Dataset
from models.select_model import define_Model

import cv2
import zipfile


def main(json_path='options/test.json'):

    '''
    # ----------------------------------------
    # Step--1 (prepare opt)
    # ----------------------------------------
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default=json_path, help='Path to option JSON file.')
    parser.add_argument('--launcher', default='pytorch', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--dist', default=False)
    parser.add_argument('--dataPath', type=str,help='dataPath of low resolution images')
    parser.add_argument('--savePath', type=str,help='save path for upscaled images')
    parser.add_argument('--weightPath', type=str,help='weight path')
    parser.add_argument("--modelVersion", default=[], nargs='+', type=str)

    opt = option.parse(parser.parse_args().opt, is_train=False)

    # ----------------------------------------
    # distributed settings
    # ----------------------------------------
    if opt['dist']:
        init_dist('pytorch')
    opt['rank'], opt['world_size'] = get_dist_info()

    if opt['rank'] == 0:
        util.mkdirs((path for key, path in opt['path'].items() if 'pretrained' not in key))

    border = opt['scale']
    # --<--<--<--<--<--<--<--<--<--<--<--<--<-

    # ----------------------------------------
    # save opt to  a '../option.json' file
    # ----------------------------------------
    if opt['rank'] == 0:
        option.save(opt)

    # ----------------------------------------
    # return None for missing key
    # ----------------------------------------
    opt = option.dict_to_nonedict(opt)


    '''
    # ----------------------------------------
    # Step--2 (creat dataloader)
    # ----------------------------------------
    '''

    # ----------------------------------------
    # 1) create_dataset
    # 2) creat_dataloader for train and test
    # ----------------------------------------

    opt['datasets']['test']['dataroot_L'] = parser.parse_args().dataPath
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'test':
            test_set = define_Dataset(dataset_opt)
            test_loader = DataLoader(test_set, batch_size=1,
                                     shuffle=False, num_workers=1,
                                     drop_last=False, pin_memory=True)
        elif phase == 'train':
            pass
        else:
            raise NotImplementedError("Phase [%s] is not recognized." % phase)

    '''
    # ----------------------------------------
    # Step--3 (initialize model)
    # ----------------------------------------
    '''
    for idx in range(len(parser.parse_args().modelVersion)):
        print("---------------------------------------------------------------------")
        print("%s.pth model test start"%(parser.parse_args().modelVersion[idx]))

        opt['path']['pretrained_netG'] = "{}{}_G.pth".format(parser.parse_args().weightPath,parser.parse_args().modelVersion[idx])
        opt['path']['pretrained_netE'] = "{}{}_E.pth".format(parser.parse_args().weightPath,parser.parse_args().modelVersion[idx])
        opt['path']['pretrained_optimizerG'] = "{}{}_optimizerG.pth".format(parser.parse_args().weightPath,parser.parse_args().modelVersion[idx])
        
        model = define_Model(opt)
        model.load()

        '''
        # ----------------------------------------
        # Step--4 (main testing)
        # ----------------------------------------
        '''

        for test_data in test_loader:
            image_name_ext = os.path.basename(test_data['L_path'][0])
            img_name, ext = os.path.splitext(image_name_ext)

            util.mkdir("{}/{}/".format(parser.parse_args().savePath,parser.parse_args().modelVersion[idx]))

            model.feed_data(test_data,need_H=False)
            model.test()

            visuals = model.current_visuals(need_H=False)
            E_img = util.tensor2uint(visuals['E'])
            h,w,c = E_img.shape

            # -----------------------
            # save estimated image E
            # -----------------------
            save_img_path = os.path.join("{}/{}/".format(parser.parse_args().savePath,parser.parse_args().modelVersion[idx]), '{:s}.png'.format(img_name))
            util.imsave(E_img, save_img_path)

            print("{}/{}/".format(parser.parse_args().savePath,parser.parse_args().modelVersion[idx]),'{:s}.png'.format(img_name))
    # ----------------------------------------
    # Ensemble multiple results if model is tested in several versions
    # and make zip file for submit the final result
    # ----------------------------------------

    submission = zipfile.ZipFile("{}/submission.zip".format(parser.parse_args().savePath), 'w')
    print("---------------------------------------------------------------------")
    print("Make submission zip file")

    if len(parser.parse_args().modelVersion) != 1:
        imgList = sorted(os.listdir("{}/{}/".format(parser.parse_args().savePath,parser.parse_args().modelVersion[0])))

        util.mkdir("{}/ensemble/".format(parser.parse_args().savePath))

        for imgName in imgList:
            print("%s image ensemble"%(imgName))
            result = np.zeros((h,w,c))
            for version in parser.parse_args().modelVersion:
                img = cv2.imread("{}/{}/{:s}".format(parser.parse_args().savePath,version,imgName))
                result += img
            
            result /= len(parser.parse_args().modelVersion)
            result = result.astype(np.uint8)

            cv2.imwrite("{}/ensemble/{:s}".format(parser.parse_args().savePath,imgName),result)

        imgList = sorted(os.listdir("{}/ensemble/".format(parser.parse_args().savePath)))
        os.chdir("{}/ensemble/".format(parser.parse_args().savePath))
        for imgName in imgList:
            submission.write("{:s}".format(imgName))

    else:
        imgList = sorted(os.listdir("{}/{}/".format(parser.parse_args().savePath,parser.parse_args().modelVersion[0])))
        os.chdir("{}/{}/".format(parser.parse_args().savePath,parser.parse_args().modelVersion[0]))
        for imgName in imgList:
            submission.write("{:s}".format(imgName))


    submission.close()
    print('Done.')

if __name__ == '__main__':
    main()
