'''
后处理：使用连通域进行后处理
'''

import cc3d
import nibabel as nib
from pathlib2 import Path
from tqdm import tqdm
import numpy as np
import os
import glob
import cv2
import SimpleITK as sitk

from scipy.ndimage.morphology import binary_fill_holes

from skimage.measure import label
import matplotlib.pyplot as plt

from matplotlib import pyplot as plt




def plot_test(cc_sum,name,savepath):
    x_list = []
    y_list = []

    for i in range(2,len(cc_sum),1):
        x_list.append(cc_sum[i][0])
        y_list.append(cc_sum[i][1])
    # plot_test(x_list,y_list,name)
    bg=cc_sum[0][1]
    maxconnet=cc_sum[1][1]
    plt.rcParams['font.family'] = 'SimHei'
    plt.rcParams['font.size'] = 10
    # a = y_list
    # plt.hist(a, bins=x_list)

    arrayx = np.array(x_list)
    arrayy = np.array(y_list)
    plt.bar(arrayx, arrayy)
    plt.title(name.split('.nii.gz')[0] + "_26连通域下灰度值分布\n background = "+str(bg)+'\n maxconnect = '+str(maxconnet))
    plt.xlabel('灰度值')
    plt.ylabel('像素个数')
    plt.savefig(savepath+'connect_image\\'+name.split('.nii.gz')[0] + "_26连通域下灰度值分布"+'.png')
    plt.show()


def Threshold(data, output,name):
        pred = data
        # target = target_list[N]
        vol_nii = nib.load(pred).get_fdata()
        # target_nii = nib.load(target)
        img = sitk.ReadImage(pred)
        mask=sitk.BinaryThreshold(img, lowerThreshold=0.01, upperThreshold=vol_nii.max(), insideValue=1, outsideValue=0)
        sitk.WriteImage(mask, output +name)

        # vol_image = nib.Nifti1Image(vol, affine)
        # nib.save(vol_image, output +'post_image\\'+ '26post_0.2%' + name)

def post_processing(vol, target_arr,name,affine,savepath):
    vol_ = vol.copy()###值传递     vol_1 = vol#址传递
    vol_[vol_ > 0] = 1
    vol_ = vol_.astype(np.int64)
    vol_cc = cc3d.connected_components(vol_, connectivity=26)  #### 连通域越小，分的区域就越细越多   得到所有的区域赋予不同灰度值（同一连通域相同灰度值)如何根据连通域赋不同的灰度值

    #TODO 保存连通域图像
    connect_image=nib.Nifti1Image(vol_cc,affine)
    nib.save(connect_image,savepath+'connect_image\\'+'26connnet_'+name)

    ##cc_sum 表示所有的连通域，i=1表示灰度值为1，vol_cc[vol_cc == i].shape[0]表示该连通域i=1有多少个即有多少个像素块
    cc_sum = [(i, vol_cc[vol_cc == i].shape[0]) for i in range(vol_cc.max() + 1)]

    cc_sum.sort(key=lambda x: x[1], reverse=True)  ###排序，按照连通域的数量
    plot_test(cc_sum,name,savepath)#绘制灰度缝补plot 图像并保存

    cc_sum.pop(0)  # remove background
    save_reduce_cc=False

    if save_reduce_cc:
        reduce_cc = [cc_sum[i][0] for i in range(0, len(cc_sum)) if cc_sum[i][1] >=cc_sum[0][1] * 0.002]
        for i in reduce_cc:
            vol[vol_cc == i] = 0  #######对于reduce_cc的区域赋值0 实现删除其他区域 保留最大连通域
    else:
    ##筛选出需要移除的连通域
        reduce_cc = [cc_sum[i][0] for i in range(1, len(cc_sum)) if cc_sum[i][1] < cc_sum[0][1] * 0.002]
        for i in reduce_cc:
            vol[vol_cc == i] = 0  #######对于reduce_cc的区域赋值0 实现删除其他区域 保留最大连通域
    return vol
def grow_threshold(grayImg,output,name):
    arr = nib.load(grayImg).get_fdata()
    outImg = np.zeros(arr.shape)  # 建立一个同样大小的空矩阵
    w,h, d = arr.shape
    for i in range(w):
        for j in range(h):
            for k in range(d):
                data = arr[i, j, k]
                if (data>=500):#and(k!=30)and(k!=54)and(k!=78)
                    outImg[i, j, k] = 1
    # print('outImg shape ',outImg)
    outImg=np.array(outImg,dtype='uint8')
    outImg = nib.Nifti1Image(outImg,nib.load(grayImg).affine)
    nib.save(outImg, output + 't_02' + name)  # 6x6x6_0.82_500
# def hole_filling(bw, hole_min, hole_max, fill_2d=True):
#     bw = bw > 0
#     if len(bw.shape) == 2:
#         background_lab = label(~bw, connectivity=1)
#         fill_out = np.copy(background_lab)
#         component_sizes = np.bincount(background_lab.ravel())
#         too_big = component_sizes > hole_max
#         too_big_mask = too_big[background_lab]
#         fill_out[too_big_mask] = 0
#         too_small = component_sizes < hole_min
#         too_small_mask = too_small[background_lab]
#         fill_out[too_small_mask] = 0
#     elif len(bw.shape) == 3:
#         if fill_2d:
#             fill_out = np.zeros_like(bw)
#             for zz in range(bw.shape[1]):
#                 background_lab = label(~bw[:, zz, :], connectivity=1)  # 1表示4连通， ~bw[zz, :, :]1变为0， 0变为1
#                 # 标记背景和孔洞， target区域标记为0
#                 out = np.copy(background_lab)
#                 # plt.imshow(bw[:, :, 87])
#                 # plt.show()
#                 component_sizes = np.bincount(background_lab.ravel())  # ravel()方法将数组维度拉成一维数组
#                 # 求各个类别的个数
#                 too_big = component_sizes > hole_max
#                 too_big_mask = too_big[background_lab]
#
#                 out[too_big_mask] = 0
#
#                 too_small = component_sizes < hole_min
#                 too_small_mask = too_small[background_lab]
#                 out[too_small_mask] = 0
#                 # 大于最大孔洞和小于最小孔洞的都标记为0， 所以背景部分被标记为0了。只剩下符合规则的孔洞
#                 fill_out[:, zz, :] = out
#                 # 只有符合规则的孔洞区域是1， 背景及target都是0
#         else:
#             background_lab = label(~bw, connectivity=1)
#             fill_out = np.copy(background_lab)
#             component_sizes = np.bincount(background_lab.ravel())
#             too_big = component_sizes > hole_max
#             too_big_mask = too_big[background_lab]
#             fill_out[too_big_mask] = 0
#             too_small = component_sizes < hole_min
#             too_small_mask = too_small[background_lab]
#             fill_out[too_small_mask] = 0
#     else:
#         print('error')
#         return
#
#     return np.logical_or(bw, fill_out)  # 或运算，孔洞的地方是1，原来target的地方也是1


#
# class Point(object):
#  def __init__(self,x,y):
#   self.x = x
#   self.y = y
#
#  def getX(self):
#   return self.x
#  def getY(self):
#   return self.y
#
# def getGrayDiff(img,currentPoint,tmpPoint):
#  return abs(int(img[currentPoint.x,currentPoint.y]) - int(img[tmpPoint.x,tmpPoint.y]))
#
# def selectConnects(p):
#  if p != 0:
#   connects = [Point(-1, -1), Point(0, -1), Point(1, -1), Point(1, 0), Point(1, 1), \
#      Point(0, 1), Point(-1, 1), Point(-1, 0)]
#  else:
#   connects = [ Point(0, -1), Point(1, 0),Point(0, 1), Point(-1, 0)]
#  return connects
#
# def regionGrow(img,seeds,thresh,p = 1):
#  height, weight = img.shape
#  seedMark = np.zeros(img.shape)
#  seedList = []
#  for seed in seeds:
#   seedList.append(seed)
#  label = 1
#  connects = selectConnects(p)
#  while(len(seedList)>0):
#   currentPoint = seedList.pop(0)
#
#   seedMark[currentPoint.x,currentPoint.y] = label
#   for i in range(8):
#    tmpX = currentPoint.x + connects[i].x
#    tmpY = currentPoint.y + connects[i].y
#    if tmpX < 0 or tmpY < 0 or tmpX >= height or tmpY >= weight:
#     continue
#    grayDiff = getGrayDiff(img,currentPoint,Point(tmpX,tmpY))
#    if grayDiff < thresh and seedMark[tmpX,tmpY] == 0:
#     seedMark[tmpX,tmpY] = label
#     seedList.append(Point(tmpX,tmpY))
#  return seedMark
#
#
# img = cv2.imread('lean.png',0)
# seeds = [Point(10,10),Point(82,150),Point(20,300)]
# binaryImg = regionGrow(img,seeds,10)
# cv2.imshow(' ',binaryImg)
# cv2.waitKey(0)


if __name__ == '__main__':
    data = r'E:\Challenges2023\XunFei\Brain_PET\datasets\Pre_processing\train_all\\'  # 分割结果地址，图像为nii.gz
    output = r'E:\Challenges2023\XunFei\Brain_PET\datasets\Pre_processing\train_threshold\\'  # 移除假阳性后保存地址
    if not os.path.exists(output):
        os.makedirs(output)

    data_list = sorted(glob.glob(os.path.join(data, '*.nii.gz')))
    for N in range(0,len(data_list),1):
        print('\r[ %d / %d]' % (N, len(data_list)), end='')

        name=data_list[N].split('train_all\\')[-1]
        Threshold(data_list[N], output,name)
        # grow_threshold(data_list[N], output,name)
    # hole_filling(arr, 0, 100, fill_2d=True)
