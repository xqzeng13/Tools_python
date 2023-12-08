import SimpleITK as sitk
import cv2
import numpy as np
from matplotlib import pyplot as plt
import sys
import os
# import gdalTools
from scipy import ndimage as ndi
# from skimage.morphology import remove_small_holes, closing, square, opening, remove_small_objects, watershed
import glob as glob
import nibabel as nib
import datetime


# from tkinter
# import gui
#
# def grow(img, seed, t):
#     """
#     img: ndarray, ndim=3
#         An image volume.
#
#     seed: tuple, len=3
#         Region growing starts from this point.
#     t: int
#         The image neighborhood radius for the inclusion criteria.
#     """
#     seg = np.zeros(img.shape, dtype=np.bool)
#     checked = np.zeros_like(seg)
#
#     seg[seed] = True
#     checked[seed] = True
#     needs_check = get_nbhd(seed, checked, img.shape)
#
#     while len(needs_check) > 0:
#         pt = needs_check.pop()
#
#         # Its possible that the point was already checked and was
#         # put in the needs_check stack multiple times.
#         if checked[pt]: continue
#
#         checked[pt] = True
#
#         # Handle borders.
#         imin = max(pt[0] - t, 0)
#         imax = min(pt[0] + t, img.shape[0] - 1)
#         jmin = max(pt[1] - t, 0)
#         jmax = min(pt[1] + t, img.shape[1] - 1)
#         kmin = max(pt[2] - t, 0)
#         kmax = min(pt[2] + t, img.shape[2] - 1)
#
#         if img[pt] >= img[imin:imax + 1, jmin:jmax + 1, kmin:kmax + 1].mean():
#             # Include the voxel in the segmentation and
#             # add its neighbors to be checked.
#             seg[pt] = True
#             needs_check += get_nbhd(pt, checked, img.shape)
#
#     return seg
# #
# # # -*- coding:utf-8 -*-
#
# # #
# # # import matplotlib.pyplot as plt
# # # ####################################################################################
# # #
# # #
# # # #######################################################################################
# # # class Point(object):
# # #     def __init__(self, x, y):
# # #         self.x = x
# # #         self.y = y
# # #
# # #     def getX(self):
# # #         return self.x
# # #
# # #     def getY(self):
# # #         return self.y
# # #
# # #
# # # connects = [Point(-1, -1), Point(0, -1), Point(1, -1), Point(1, 0), Point(1, 1), Point(0, 1), Point(-1, 1),
# # #             Point(-1, 0)]
# # #
# # #
# # # #####################################################################################
# # # # 计算两个点间的欧式距离
# # # def get_dist(image, seed_location1, seed_location2):
# # #     l1 = image[seed_location1.x, seed_location1.y]
# # #     l2 = image[seed_location2.x, seed_location2.y]
# # #     count = np.sqrt(np.sum(np.square(l1 - l2)))
# # #     return count
# # #
# # #
# # # ########################################################################################
# # # def reginGrow(image, mask):
# # #     im_shape = image.shape
# # #     height = im_shape[0]
# # #     width = im_shape[1]
# # #
# # #     markers = ndi.label(mask, output=np.uint32)[0]
# # #     unis = np.unique(markers)
# # #     # 获取种子点
# # #     seed_list = []
# # #
# # #     for uni in unis:
# # #         if uni == 0:
# # #             continue
# # #         pointsX, pointsY = np.where(markers == uni)
# # #         num_point = len(pointsX) // 4
# # #         for i in [0, num_point * 1, num_point * 2, num_point * 3]:
# # #             pointX, pointY = pointsX[i], pointsY[i]
# # #             seed_list.append(Point(pointX, pointY))
# # #
# # #     # 标记，判断种子是否已经生长
# # #     img_mark = np.zeros([height, width])
# # #
# # #     T = 7.5  # 阈值
# # #     class_k = 1  # 类别
# # #     # 生长一个类
# # #     while (len(seed_list) > 0):
# # #         seed_tmp = seed_list[0]
# # #         # 将以生长的点从一个类的种子点列表中删除
# # #         seed_list.pop(0)
# # #
# # #         img_mark[seed_tmp.x, seed_tmp.y] = class_k
# # #
# # #         # 遍历8邻域
# # #         for i in range(8):
# # #             tmpX = seed_tmp.x + connects[i].x
# # #             tmpY = seed_tmp.y + connects[i].y
# # #
# # #             if (tmpX < 0 or tmpY < 0 or tmpX >= height or tmpY >= width):
# # #                 continue
# # #             dist = get_dist(image, seed_tmp, Point(tmpX, tmpY))
# # #             # 在种子集合中满足条件的点进行生长
# # #             if (dist < T and img_mark[tmpX, tmpY] == 0):
# # #                 img_mark[tmpX, tmpY] = class_k
# # #                 seed_list.append(Point(tmpX, tmpY))
# # #     img_mark = img_mark + mask
# # #     img_mark = remove_small_holes(img_mark.astype(np.uint8), 100)
# # #     return np.where(img_mark > 0, 1, 0)
# # #
# # #
# # # if __name__ == '__main__':
# # #
# # #     #import Image
# # #     im_proj, im_geotrans, im_width, im_height, im = gdalTools.read_img('data/image.tif')
# # #     im = im.transpose((1, 2, 0))
# # #     image = im.copy()
# # #     _, _, _, _, mask = gdalTools.read_img('data/seed.tif')
# # #     img_mark = reginGrow(image, mask)
# # import numpy as np
# # import cv2
# #
# # # class Point(object):
# # #     def __init__(self,x,y):
# # #         self.x = x
# # #         self.y = y
# # #
# # #     def getX(self):
# # #         return self.x
# # #     def getY(self):
# # #         return self.y
# # #
# # # def getGrayDiff(img,currentPoint,tmpPoint):
# # #      return abs(int(img[currentPoint.x,currentPoint.y]) - int(img[tmpPoint.x,tmpPoint.y]))
# # #
# # # def selectConnects(p):
# # #      if p != 0:
# # #         connects = [Point(-1, -1), Point(0, -1), Point(1, -1), Point(1, 0), Point(1, 1),Point(0, 1), Point(-1, 1), Point(-1, 0)]
# # #      else:
# # #         connects = [ Point(0, -1), Point(1, 0),Point(0, 1), Point(-1, 0)]
# # #      return connects
# # #
# # # def regionGrow(img,seeds,thresh,p = 1):
# # #      height, weight ,CH = img.shape
# # #      seedMark = np.zeros(img.shape)
# # #      seedList = []
# # #      for seed in seeds:
# # #       seedList.append(seed)
# # #      label = 1
# # #      connects = selectConnects(p)
# # #      while(len(seedList)>0):
# # #       currentPoint = seedList.pop(0)
# # #
# # #       seedMark[currentPoint.x,currentPoint.y] = label
# # #       for i in range(8):
# # #        tmpX = currentPoint.x + connects[i].x
# # #        tmpY = currentPoint.y + connects[i].y
# # #        if tmpX < 0 or tmpY < 0 or tmpX >= height or tmpY >= weight:
# # #         continue
# # #        grayDiff = getGrayDiff(img,currentPoint,Point(tmpX,tmpY))
# # #        if grayDiff < thresh and seedMark[tmpX,tmpY] == 0:
# # #         seedMark[tmpX,tmpY] = label
# # #         seedList.append(Point(tmpX,tmpY))
# # #      return seedMark
# # #
# # # if __name__ == '__main__':
# # #     img = cv2.imread(r'D:\Work\Datasets\test.png')
# # #     seeds = [Point(10,10),Point(82,150),Point(20,300)]
# # #     binaryImg = regionGrow(img,seeds,10)
# # #     cv2.imshow(' ',binaryImg)
# # #     cv2.waitKey(0)
# # # 区域生长 programmed by changhao
# # from PIL import Image
# # import matplotlib.pyplot as plt  # plt 用于显示图片
# # # import numpy as np
# #
# # im = Image.open(r'D:\Work\Datasets\test.png') # 读取图片
# # # im.show()
# #
# # im_array = np.array(im)
# #
# # # print(im_array)
# # [m, n] = im_array.shape
# #
# # a = np.zeros((m, n))  # 建立等大小空矩阵
# # a[70, 70] = 1  # 设立种子点
# # k = 40  # 设立区域判断生长阈值
# #
# # flag = 1  # 设立是否判断的小红旗
# # while flag == 1:
# #     flag = 0
# #     lim = (np.cumsum(im_array * a)[-1]) / (np.cumsum(a)[-1])
# #     for i in range(2, m):
# #         for j in range(2, n):
# #             if a[i, j] == 1:
# #                 for x in range(-1, 2):
# #                     for y in range(-1, 2):
# #                         if a[i + x, j + y] == 0:
# #                             if (abs(im_array[i + x, j + y] - lim) <= k):
# #                                 flag = 1
# #                                 a[i + x, j + y] = 1
# #
# # data = im_array * a  # 矩阵相乘获取生长图像的矩阵
# # new_im = Image.fromarray(data)  # data矩阵转化为二维图片
# #
# # # if new_im.mode == 'F':
# # #    new_im = new_im.convert('RGB')
# # # new_im.save('new_001.png') #保存PIL图片
# #
# # # 画图展示
# # plt.subplot(1, 2, 1)
# # plt.imshow(im, cmap='gray')
# # plt.axis('off')  # 不显示坐标轴
# # plt.show()
# #
# # plt.subplot(1, 2, 2)
# # plt.imshow(new_im, cmap='gray')
# # plt.axis('off')  # 不显示坐标轴
# # plt.show()
#
# def on_mouse(event, x, y, flags, params):
#     if event == cv.CV_EVENT_LBUTTONDOWN:
#         print 'Start Mouse Position: ' + str(x) + ', ' + str(y)
#         s_box = x, y
#         boxes.append(s_box)
# def region_growing(img, seed):
#     #Parameters for region growing
#     neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
#     region_threshold = 0.2
#     region_size = 1
#     intensity_difference = 0
#     neighbor_points_list = []
#     neighbor_intensity_list = []
#
#     #Mean of the segmented region
#     region_mean = img[seed]
#
#     #Input image parameters
#     height, width = img.shape
#     image_size = height * width
#
#     #Initialize segmented output image
#     segmented_img = np.zeros((height, width, 1), np.uint8)
#
#     #Region growing until intensity difference becomes greater than certain threshold
#     while (intensity_difference < region_threshold) & (region_size < image_size):
#         #Loop through neighbor pixels
#         for i in range(4):
#             #Compute the neighbor pixel position
#             x_new = seed[0] + neighbors[i][0]
#             y_new = seed[1] + neighbors[i][1]
#
#             #Boundary Condition - check if the coordinates are inside the image
#             check_inside = (x_new >= 0) & (y_new >= 0) & (x_new < height) & (y_new < width)
#
#             #Add neighbor if inside and not already in segmented_img
#             if check_inside:
#                 if segmented_img[x_new, y_new] == 0:
#                     neighbor_points_list.append([x_new, y_new])
#                     neighbor_intensity_list.append(img[x_new, y_new])
#                     segmented_img[x_new, y_new] = 255
#
#         #Add pixel with intensity nearest to the mean to the region
#         distance = abs(neighbor_intensity_list-region_mean)
#         pixel_distance = min(distance)
#         index = np.where(distance == pixel_distance)[0][0]
#         segmented_img[seed[0], seed[1]] = 255
#         region_size += 1
#
#         #New region mean
#         region_mean = (region_mean*region_size + neighbor_intensity_list[index])/(region_size+1)
#
#         #Update the seed value
#         seed = neighbor_points_list[index]
#         #Remove the value from the neighborhood lists
#         neighbor_intensity_list[index] = neighbor_intensity_list[-1]
#         neighbor_points_list[index] = neighbor_points_list[-1]
#
#     return segmented_img
# def regionGrowing3D(grayImg, seedlist, size,target_arr):
#
#     outImg = np.zeros_like(grayImg)  # 建立一个同样大小的空矩阵
#     pointsNum = 1
#     seedlist_0=seedlist
#     ##Growth  ROI 区域     去除在大静脉生长情况
#     # x_min, x_max, y_min, y_max, z_min, z_max = get_ROI_area(target_arr)
#     for N in range(0,len(seedlist),1):
#
#         seed = seedlist[N]
#         # seed=(161,340,76)
#         if (seed[0] > 440) | (seed[0] <8) | (seed[1] > 440) | (seed[1] < 8) | (seed[2] > 115) | (seed[2] < 8):
#             # print(seed)
#             continue
#         # threshold = get_threshold(grayImg, seed, size)
#         # if threshold <= 150:
#         #     continue
#         # # SEED ROI 区域
#         # if seed[2] > 90:  # 层数过大，seed在静脉处z
#         #     continue
#         # if seed[1] < 100 & seed[1] > 350:  # y
#         #     continue
#         # if 250 > seed[0] > 180:  # x
#         #     continue
#         [maxX, maxY, maxZ] = grayImg.shape
#         # 用于保存生长点的队列
#         seedlist_1 = []
#         seedlist_2=[]
#         seedlist_1.append((seed[0], seed[1], seed[2]))
#         seedlist_2.append((seed[0], seed[1], seed[2]))
#         outImg[seed[0], seed[1], seed[2]] = 1  # 从种子点开始值设为1；
#         # 初始生长的数
#         # pointsMean = float(grayImg[seed[0], seed[1], seed[2]])  # 该点在原图像的像素值
#
#         # 用于计算生长点周围26个点的位置
#         # Next26 = [[-1, -1, -1],[-1, 0, -1],[-1, 1, -1],
#         #             [-1, 1, 0], [-1, -1, 0], [-1, -1, 1],
#         #             [-1, 0, 1], [-1, 0, 0],[-1, 0, -1],
#         #             [0, -1, -1], [0, 0, -1], [0, 1, -1],
#         #             [0, 1, 0],[-1, 0, -1],
#         #             [0, -1, 0],[0, -1, 1],[-1, 0, -1],
#         #             [0, 0, 1],[1, 1, 1],[1, 1, -1],
#         #             [1, 1, 0],[1, 0, 1],[1, 0, -1],
#         #             [1, -1, 0],[1, 0, 0],[1, -1, -1]]
#         Next26 = [[-1, -1, -1], [-1, 0, -1], [-1, 1, -1],
#                   [-1, 1, 0], [-1, -1, 0], [-1, -1, 1],
#                   [-1, 0, 1], [-1, 0, 0], [-1, 1, 1],
#                   [0, 1, -1], [0, 1, 1], [0, 1, -0],
#                   [0, 0, 1], [0, 0, -1],
#                   [0, -1, 0], [0, -1, 1], [0, -1, -1],
#                   [1, 0, 1], [1, 0, -1], [1, 0, 0],
#                   [1, 1, 0], [1, 1, 1], [1, 1, -1],
#                   [1, -1, 0], [1, -1, 1], [1, -1, -1]]
#         Next26list=[]
#
#
#
#         Next18 = [
#                   [-1, 1, 0], [-1, -1, 0], [-1, -1, 1],
#                   [-1, 0, 1], [-1, 0, 0], [-1, 1, 1],
#                    [0, 1, 1], [0, 1, -0],
#                   [0, 0, 1], [0, 0, -1],
#                   [0, -1, 0], [0, -1, 1],
#                   [1, 0, 1], [1, 0, 0],
#                   [1, 1, 0], [1, 1, 1],
#                   [1, -1, 0], [1, -1, 1]]
#         growthtime = 0
#
#
#         while (len(seedlist_1) > 0):
#             # if (len(pointQueue) > 0):
#             # 取出队首并删除
#             growSeed = seedlist_1[0]
#             if (growSeed[0] > 440) | (growSeed[0] < 8) | (growSeed[1] > 440) | (growSeed[1] < 8) | (growSeed[2] > 115) | (growSeed[2] < 8):
#                 # print(seed)
#                 continue
#             max,mean,std = get_threshold(grayImg, growSeed, size)
#             ####筛选种子点
#             if (max <= 150)or(std<20):#mean std 限制噪声引起的误差可结合对应点灰度值限制or (max >= 300)or(mean>100)or (std<60)or(std<40)or(std<20)
#                 del seedlist_1[0]#阈值小于150的点肯定不是血管,大于300也不是所需血管将该种子点
#                 continue
#             del seedlist_1[0]
#
#             # for differ in Next26:
#             #     growPointx = growSeed[0] + differ[0]
#             #     growPointy = growSeed[1] + differ[1]
#             #     growPointz = growSeed[2] + differ[2]
#             #     area = (growPointx, growPointy, growPointz)
#             #     Next26list.append(area)
#
#             for differ in Next26:
#                 growPointx = growSeed[0] + differ[0]
#                 growPointy = growSeed[1] + differ[1]
#                 growPointz = growSeed[2] + differ[2]
#                 # 是否是边缘点
#                 if ((growPointx < 0) or (growPointx > maxX - 1) or
#                         (growPointy < 0) or (growPointy > maxY - 1) or (growPointz < 0) or (growPointz > maxZ - 1)):
#                     continue
#                 # 是否已经被生长
#                 if (outImg[growPointx, growPointy, growPointz] == 1):
#                     # if (outImg[growPointx, growPointy, growPointz] == 255):
#                     continue
#                 data = grayImg[growPointx, growPointy, growPointz]
#                 # z=[30,54,78]#伪影严重区域
#                 # if ( growPointz in set(z)):#(250 > growPointx > 180)or
#                 #     zz=0
#                 if data>=170:
#                     outImg[growPointx, growPointy, growPointz] = 1
#                     seedlist_1.append((growPointx, growPointy, growPointz))
#                     seedlist_2.append((growPointx, growPointy, growPointz))
#                 # 判断条件
#                 # 符合条件则生长，并且加入到生长点队列中
#                 ##阈值的选取怎么样才更合理
#                 if growthtime >= 500:
#                     break
#                 if (data >= 0.8*max)and (data>=160):#（双阈值max*0.85,1.15*max）and(data>=150)and (data<300)1.3*threshold>=  and (data>=170)and (data>=160
#                     pointsNum += 1
#                     ##Growth  ROI 区域     去除在大静脉生长情况
#                     # x_min, x_max, y_min, y_max, z_min, z_max = get_ROI_area(target_arr)
#                     # if growPointz >80:  # 层数过大，seed在静脉处z
#                     #     continue
#                     # if growPointy < 105 & growPointy > 350:  # y
#                     #     continue
#                     if 250 > growPointx > 180:  # x
#                         continue
#
#                     outImg[growPointx, growPointy, growPointz] = 1
#                     seedlist_2se=set(seedlist_2)
#                     seedlist_0se=set(seedlist_0)
#                     #判断该点是否在种子点和次种子点栈里面，若在则不加入种子点中，不在则加入为新的种子点栈中
#                     if ((growPointx, growPointy, growPointz)in seedlist_2se)or((growPointx, growPointy, growPointz)in seedlist_0se):
#                         continue
#                     else:
#                         seedlist_1.append((growPointx, growPointy, growPointz))
#                         seedlist_2.append((growPointx, growPointy, growPointz))
#
#                         growthtime += 1
#
#
#                 # if(abs(data - pointsMean)<30):
#                 #     pointsNum += 1
#                 #     # pointsMean = (pointsMean * (pointsNum - 1) + data) / pointsNum
#                 #     pointsMean = data
#                 #
#                 #     outImg[growPointx, growPointy,growPointz] = 1
#                 #     pointQueue.append([growPointx, growPointy,growPointz])
#     return outImg
def regionGrowing3D(grayImg, seedlist, size,target_arr):

    outImg = np.zeros_like(grayImg)  # 建立一个同样大小的空矩阵
    pointsNum = 1
    seedlist_2 = []
    seedlist_0=seedlist
    ##Growth  ROI 区域     去除在大静脉生长情况
    # x_min, x_max, y_min, y_max, z_min, z_max = get_ROI_area(target_arr)
    for N in range(0,len(seedlist),1):
        # if N >=7609:
        #     continue
        seed = seedlist[N]
        # seed=(161,340,76)
        if (seed[0] > 440) | (seed[0] <8) | (seed[1] > 440) | (seed[1] < 8) | (seed[2] > 115) | (seed[2] < 8):
            # print(seed)
            continue
        # threshold = get_threshold(grayImg, seed, size)
        # if threshold <= 150:
        #     continue
        # # SEED ROI 区域
        # if seed[2] > 90:  # 层数过大，seed在静脉处z
        #     continue
        # if seed[1] < 100 & seed[1] > 350:  # y
        #     continue
        # if 250 > seed[0] > 180:  # x
        #     continue
        [maxX, maxY, maxZ] = grayImg.shape
        # 用于保存生长点的队列
        seedlist_1 = []

        seedlist_1.append((seed[0], seed[1], seed[2]))
        seedlist_2.append((seed[0], seed[1], seed[2]))
        outImg[seed[0], seed[1], seed[2]] = 1  # 从种子点开始值设为1；
        # 初始生长的数
        # pointsMean = float(grayImg[seed[0], seed[1], seed[2]])  # 该点在原图像的像素值

        # 用于计算生长点周围26个点的位置
        # Next26 = [[-1, -1, -1],[-1, 0, -1],[-1, 1, -1],
        #             [-1, 1, 0], [-1, -1, 0], [-1, -1, 1],
        #             [-1, 0, 1], [-1, 0, 0],[-1, 0, -1],
        #             [0, -1, -1], [0, 0, -1], [0, 1, -1],
        #             [0, 1, 0],[-1, 0, -1],
        #             [0, -1, 0],[0, -1, 1],[-1, 0, -1],
        #             [0, 0, 1],[1, 1, 1],[1, 1, -1],
        #             [1, 1, 0],[1, 0, 1],[1, 0, -1],
        #             [1, -1, 0],[1, 0, 0],[1, -1, -1]]
        Next26 = [[-1, -1, -1], [-1, 0, -1], [-1, 1, -1],
                  [-1, 1, 0], [-1, -1, 0], [-1, -1, 1],
                  [-1, 0, 1], [-1, 0, 0], [-1, 1, 1],
                  [0, 1, -1], [0, 1, 1], [0, 1, -0],
                  [0, 0, 1], [0, 0, -1],
                  [0, -1, 0], [0, -1, 1], [0, -1, -1],
                  [1, 0, 1], [1, 0, -1], [1, 0, 0],
                  [1, 1, 0], [1, 1, 1], [1, 1, -1],
                  [1, -1, 0], [1, -1, 1], [1, -1, -1]]
        Next26list=[]
        Next18 = [
                  [-1, 1, 0], [-1, -1, 0], [-1, -1, 1],
                  [-1, 0, 1], [-1, 0, 0], [-1, 1, 1],
                   [0, 1, 1], [0, 1, -0],
                  [0, 0, 1], [0, 0, -1],
                  [0, -1, 0], [0, -1, 1],
                  [1, 0, 1], [1, 0, 0],
                  [1, 1, 0], [1, 1, 1],
                  [1, -1, 0], [1, -1, 1]]

        while (500>len(seedlist_1) > 0):
            growSeed = seedlist_1[0]
            ####筛选种子点
            data = grayImg[growSeed[0], growSeed[1], growSeed[2]]

            max,mean,std = get_threshold(grayImg, growSeed, size)
            if (data <= 160)or((mean/std)>5):#mean std 限制噪声引起的误差可结合对应点灰度值限制or (max >= 300)or(mean>100)or (std<60)or(std<25)
                del seedlist_1[0]#阈值小于150的点肯定不是血管,大于300也不是所需血管将该种子点
                continue
            del seedlist_1[0]
            growthtime = 0 #一个次种子点生长次数
            for differ in Next26:
                growPointx = growSeed[0] + differ[0]
                growPointy = growSeed[1] + differ[1]
                growPointz = growSeed[2] + differ[2]
                # 是否是边缘点
                if ((growPointx < 0) or (growPointx > maxX - 1) or
                        (growPointy < 0) or (growPointy > maxY - 1) or (growPointz < 0) or (growPointz > maxZ - 1)):
                    continue
                # 是否已经被生长
                if (outImg[growPointx, growPointy, growPointz] == 1):
                    # if (outImg[growPointx, growPointy, growPointz] == 255):
                    continue
                data = grayImg[growPointx, growPointy, growPointz]###data都大于195因为阈值设的195
                # 判断条件
                # 符合条件则生长，并且加入到生长点队列中
                ##阈值的选取怎么样才更合理
                if (data >= 0.65*max):#（双阈值max*0.85,1.15*max）and(data>=150)and (data<300)1.3*threshold>=  and (data>=170)and (data>=160
                    # pointsNum += 1
                    ##Growth  ROI 区域     去除在大静脉生长情况
                    # x_min, x_max, y_min, y_max, z_min, z_max = get_ROI_area(target_arr)
                    # if growPointz >80:  # 层数过大，seed在静脉处z
                    #     continue
                    # if growPointy < 105 & growPointy > 350:  # y
                    #     continue
                    if 275 > growPointx > 180:  # x
                        continue

                    outImg[growPointx, growPointy, growPointz] = 1
                    seedlist_2se=set(seedlist_2)
                    seedlist_0se=set(seedlist_0)
                    #判断该点是否在种子点和次种子点栈里面，若在则不加入种子点中，不在则加入为新的种子点栈中
                    if ((growPointx, growPointy, growPointz)in seedlist_2se)or((growPointx, growPointy, growPointz)in seedlist_0se):
                        continue
                    else:
                        seedlist_1.append((growPointx, growPointy, growPointz))
                        seedlist_2.append((growPointx, growPointy, growPointz))

                        growthtime += 1
                        # print(growthtime)
                        if growthtime >=500:
                            break

                # if(abs(data - pointsMean)<30):
                #     pointsNum += 1
                #     # pointsMean = (pointsMean * (pointsNum - 1) + data) / pointsNum
                #     pointsMean = data
                #
                #     outImg[growPointx, growPointy,growPointz] = 1
                #     pointQueue.append([growPointx, growPointy,growPointz])
    return outImg
def get_seed(label):
    label = nib.load(label).get_fdata()
    print('seed sum: ',label.sum())
    C, K, H = label.shape
    seedlist = []
    for i in range(C):
        for j in range(K):
            for k in range(H):
                if label[i][j][k] != 0:
                    posite = (i, j, k)  # 对应itk-snap 值减1，itk-snap是从1开始，而数组从0开始
                    seedlist.append(posite)
    print('get seeds is finished! ')

    return seedlist
def get_threshold(image_arr, seed, size):
    x, y, z = seed
    position_value = []
    for i in range(int(x - size[0] / 2), int(x + size[0] / 2), 1):  ##向上取整
        for j in range(int(y - size[1] / 2), int(y + size[1] / 2), 1):
            for k in range(int(z - size[2] / 2), int(z + size[2] / 2), 1):
                pos = (i, j, k)
                position_value.append(image_arr[pos])

    position_value_arr = np.array(position_value)
    mean=position_value_arr.mean()#
    std=position_value_arr.std()
    # max=position_value_arr.max()
    # thresholdup=mean+std
    # thresholdlow=mean-std

    return position_value_arr.max(),mean,std
def get_ROI_area(target):
    assert len(target.shape) == 3
    z_min = 0
    y_min = 0
    x_min = 0
    z_max = 0
    y_max = 0
    x_max = 0

        # TODO Y
    for x in range(target.shape[0]):
        if target[x, :, :].sum() != 0:
            x_min = x
            # print("a_min is :", x)
            break
    for x in range(target.shape[0]-1, x_min, 1):
        if target[x, :, :].sum() == 0:
            x_max = x
            # print("a_min is :", x)
            break
        #TODO Y
    for y in range(target.shape[1]):
        if target[:, y, :].sum() != 0:
            y_min = y
            break

    for y in range(target.shape[1]-1, y_min, -1):
        if target[:, y, :].sum() == 0:
            y_max = y
            # print("a_min is :", x)
            break
        # TODO Y
    for z in range(target.shape[2]):
        if target[:, :, z].sum() != 0:
            z_min = z
            # print("a_min is :", x)
            break
    for z in range(target.shape[2]-1,z_min , 1):
        if target[:, :, z].sum() == 0:
            z_max = z
            # print("a_min is :", x)
            break
    return x_min,x_max,y_min,y_max,z_min,z_max
def sitkregionGrowing(image, seed):

    initial_seed_point_indexes = [(132, 142, 96)]

    mask = sitk.ConnectedThreshold(image, seedlist=initial_seed_point_indexes, lower=300, upper=380)

if __name__ == '__main__':
    datapath = r'D:\Work\Datasets\Normaldatas89\\'
    savepath = r'D:\Work\Datasets\Normaldatas89\areagrow\label_grow\\'
    seedsave = r'D:\Work\Datasets\Normaldatas89\areagrow\seed_grow\\'

    imglist = sorted(glob.glob(os.path.join(datapath, "dataimage\\" + "*.nii.gz")))
    GTlist = sorted(glob.glob(os.path.join(datapath, "threshold\connect\seed\\" + "t_02*")))
    seedlist = sorted(glob.glob(os.path.join(datapath, "threshold\connect\seed\\" + "t_02*")))
    size = (6,6,6)  ##必须为 int 偶数 目前10最合理

    for num in range(0,3, 1):#len(imglist)
        start_time = datetime.datetime.now()
        print('\r[ %d / %d]' % (num, len(imglist)), end='')
        name = imglist[num].split('dataimage\\')[-1]
        print(' file is : ', name)

        image = nib.load(imglist[num])
        # image = nib.load(r'E:\Improtant file\patient\noskull\12463920180529.nii.gz')

        image_arr = image.get_fdata()
        target_arr = nib.load(GTlist[num]).get_fdata()
        seed = get_seed(seedlist[num])
        # seed = get_seed(r'E:\Improtant file\patient\pred\fusion\12463920180529_fusion.nii.gz')

        mask = regionGrowing3D(image_arr, seed, size,target_arr)
        print('\n mask grow sum ',mask.sum())

        #todo fusion target&grow
        fusion_data = target_arr + mask
        fusion_data[fusion_data < 1] = 0
        fusion_data[fusion_data >= 1] = 1
        fusion_data = np.array(fusion_data, dtype='uint8')
        # 种子生长情况
        vol_image = nib.Nifti1Image(mask,image.affine)
        nib.save(vol_image,seedsave+'thre_grow02'+name)#6x6x6_0.82_500
        #综合生长情况
        fusion_data = nib.Nifti1Image(fusion_data, image.affine)
        nib.save(fusion_data, savepath+'thre_grow02'+name)  # 6x6x6_0.82_500
        end_time = datetime.datetime.now()
        print('growth time   is :', end_time - start_time)
