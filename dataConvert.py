import os 
import cv2
import numpy as np

def get_random_crop(lr,hr, crop_height, crop_width):


    max_x = lr.shape[1] - crop_width
    max_y = lr.shape[0] - crop_height

    x = np.random.randint(0, max_x)
    y = np.random.randint(0, max_y)

    lrCrop = lr[y: y + crop_height, x: x + crop_width]
    hrCrop = hr[y*4: (y + crop_height)*4, 4*x: (x + crop_width)*4]

    return lrCrop, hrCrop

 
mode = "train/"
basePath = "./originalDataset/"
savePath = "./croppedDataset/"
cropSize = 128
cropInterval = 32

dataType = ['lr/','hr/']

if not os.path.exists(savePath + dataType[0]):
    os.makedirs(savePath + dataType[0])
if not os.path.exists(savePath + dataType[1]):
    os.makedirs(savePath + dataType[1])

imageList = os.listdir(basePath + mode + dataType[0])

count = 0

for imgPath in imageList:
    lrImage = cv2.imread(basePath + mode + dataType[0] + imgPath)
    hrImage = cv2.imread(basePath + mode + dataType[1] + imgPath)
    
    for h in range(int(512/cropInterval)-3):
        for w in range(int(512/cropInterval)-3):

            lrCrop = lrImage[cropInterval * h: (cropInterval * h) + cropSize, cropInterval * w: (cropInterval * w) + cropSize]
            hrCrop = hrImage[(cropInterval * h)*4: ((cropInterval * h) + cropSize)*4, (cropInterval * w)*4: ((cropInterval * w) + cropSize)*4]
            count += 1
            print(count,h,w,imgPath,lrCrop.shape,hrCrop.shape)
            cv2.imwrite(savePath + "lr/interval_"+ str(h) + "_" + str(w) + '_' + imgPath,lrCrop)
            cv2.imwrite(savePath + "hr/interval_"+ str(h) + "_" + str(w) + '_' + imgPath,hrCrop)


#randomCrop
for iter in range(100):
    i = 0
    for imgPath in imageList:
        lrImage = cv2.imread(basePath + mode + dataType[0] + imgPath)
        hrImage = cv2.imread(basePath + mode + dataType[1] + imgPath)

        lrCrop,hrCrop = get_random_crop(lrImage,hrImage,cropSize,cropSize)
        
        cv2.imwrite(savePath + "lr/rancodCrop_"+ str(iter) + "_" + imgPath,lrCrop)
        cv2.imwrite(savePath + "hr/rancodCrop_"+ str(iter) + "_" + imgPath,hrCrop)

        print("%d iter / %s images"%(iter,imgPath))


