import os.path

import SimpleITK
from tqdm import tqdm
import glob as glob
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import SimpleITK as sitk
##TODO label to nii.gz file
# path=r'E:\Challenges2023\XunFei\oct\OCT_DATA\train\labels\\'
# label_ls=sorted(glob.glob(os.path.join(path,'*.npy')))
# for label in tqdm (label_ls,total=len(label_ls)):
#     name=label.split('labels\\')[-1].replace('.npy','')
#     label_arr=np.load(label)
#     # print(label_arr.shape)
#
#     nifti_image = nib.Nifti1Image(label_arr, np.eye(4))
#
#     # 保存NIfTI文件
#     nib.save(nifti_image, r'E:\Challenges2023\XunFei\oct\process_oct\train\label\\'+name+'.nii.gz')
    # maX=label_arr.max()
    # if maX !=1:
    #     print(label)
    #
    #     plt.imshow(label_arr)
    #     plt.show()

##TODO image/label  to nii.gz file
path=r'E:\Challenges2023\XunFei\oct\OCT_DATA\test\images\\'
image_ls=sorted(glob.glob(os.path.join(path,'*.jpg')))
for image in tqdm (image_ls,total=len(image_ls)):
    name=image.split('images\\')[-1].replace('.jpg','')
    img = sitk.ReadImage(image)
    sitk.WriteImage(img, r'E:\Challenges2023\XunFei\oct\process_oct\test\data\\' + name + '.nii.gz')
    # img_arr=sitk.GetArrayFromImage(img)
    # # new_img=img_arr[165:542,0:379,:]
    # new_img=img_arr[:, :, :]
    # new_img=img
    # img=sitk.GetImageFromArray(new_img)
    # 将jpg文件转换为nii.gz文件

    # label_path=image.replace('images','labels').replace('.jpg','.npy')
    # label=np.load(label_path)
    # if label.max()>0:
    # # new_label=label[165:542, 0:379]
    #     lab=sitk.GetImageFromArray(label)
    #
    #     sitk.WriteImage(img,r'E:\Challenges2023\XunFei\oct\process_oct\train\data1\\'+name+'.nii.gz')
    #
    #     sitk.WriteImage(lab,r'E:\Challenges2023\XunFei\oct\process_oct\train\label1\\'+name+'.nii.gz')
    # label_arr=np.load(label) #IMG001-img-00001-00003.jpg
    # print(label_arr.shape)  # IMG001-img-00001-00003.npy
    #
    # nifti_image = nib.Nifti1Image(label_arr, np.eye(4))
    #
    # 保存NIfTI文件
    # nib.save(nifti_image, r'E:\Challenges2023\XunFei\oct\OCT_DATA\train\\'+name+'.nii.gz')
    # maX=label_arr.max()
    # if maX !=1:
    #     print(label)
    #
    #     plt.imshow(label_arr)
    #     plt.show()
# img=np.load(r'E:\Challenges2023\XunFei\oct\OCT_DATA\train\labels\IMG002-img-00001-00368.npy')
# # lab=nib.load(label)
# # affine=img.affine
# # img_arr=img.get_fdata()
# # # lab_arr=lab.get_fdata()
# # print(img_arr.shape)
# # arr=np.load(img)
# print(img.max(),img.mean())
